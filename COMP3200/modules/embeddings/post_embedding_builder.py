import ast
import io
import logging
import re
import time
import tokenize
from collections import namedtuple
from typing import List

from NextTagEmbedding import NextTagEmbedding, NextTagEmbeddingTrainer

from bs4 import BeautifulSoup
import spacy
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)

from custom_logger import setup_custom_logger
from unixcoder import UniXcoder

log = setup_custom_logger("post_embedding_builder", logging.INFO)


Import = namedtuple("Import", ["module", "name", "alias"])
Function = namedtuple("Function", ["function_name", "parameter_names"])


class PostEmbedding(nn.Module):
    """
    Torch module for transforming Stackoverflow posts into a torch tensor.
    """

    def __init__(self, batched=False):
        super().__init__()
        log.info("PostEmbedding instantiated!")
        self._batched = batched
        # self._global_vectors = GloVe(name='840B', dim=300)
        self._en = spacy.load("en_core_web_sm")
        self._stopwords = self._en.Defaults.stop_words
        self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._bert_model = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self._unixcoder = UniXcoder("microsoft/unixcoder-base")
        self._codebert_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )
        self._codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")

    def forward(
        self, html_batch: List[str], use_bert: bool, title_batch: List[str]
    ) -> torch.tensor:
        """
        @param html: HTML string of the body of a StackOverflow post.
        @param title: Title of a question post.
        @param flatten: Merge all paragraphs together
        @return: Post embedding represented as a torch tensor
        """

        soups = [BeautifulSoup(html, "lxml") for html in html_batch]

        assert len(soups) == len(title_batch)
        paragraph_batches = [
            self.get_paragraphs(soup, not use_bert, title)
            for soup, title in zip(soups, title_batch)
        ]

        log.debug(f"Processing {len(html_batch)} posts")

        """TIME START"""
        t1 = time.time()
        if use_bert:
            if self._batched:
                para_embs = self.to_bert_embedding(
                    [" ".join(ps) for ps in paragraph_batches]
                )
            else:
                para_embs = []
                for ps in paragraph_batches:
                    para_emb = self.to_bert_embedding([" ".join(ps)])
                    para_embs.append(torch.squeeze(para_emb))
        else:
            # para_emb = self.to_glove_paragraph_embedding(ps)
            raise NotImplementedError(
                "GloVe paragraph embedding need to be refactored to work with batches"
            )
        """TIME END"""
        t2 = time.time()
        log.debug("Function=%s, Time=%s" % ("Paragraph embedding", t2 - t1))

        code_features = [
            self.get_code(soup, get_imports_with_regex=True) for soup in soups
        ]
        modules = [x[0] for x in code_features]

        """TIME START"""
        t1 = time.time()
        if self._batched:
            code_embs = self.to_unixcode_embedding(
                [
                    "\n".join([x.get_text() for x in soup.find_all("code")])
                    for soup in soups
                ]
            )
        else:
            code_embs = []
            for soup in soups:
                code_emb = self.to_unixcode_embedding(
                    ["\n".join([x.get_text() for x in soup.find_all("code")])]
                )
                code_embs.append(torch.squeeze(code_emb))

        """TIME END"""
        t2 = time.time()
        log.debug("Function=%s, Time=%s" % ("CodeBERT embedding", t2 - t1))

        return para_embs, code_embs, modules

    def preprocess(self, text: str) -> List[str]:
        """
        @param text: Paragraph text from StackOverflow body
        @return: List of tokens which do not include stopwords, punctuation or numbers.
        """
        doc = self._en(text.lower())
        tokens = [
            word.text
            for word in doc
            if not (word.is_stop or word.is_punct or word.like_num)
        ]
        return tokens

    def get_paragraphs(
        self, soup: BeautifulSoup, preprocess: bool, title: str = None
    ) -> List[str]:
        """
        @param soup: Post body HTML wrapped in a BeautifulSoup object.
        @param title: If available, add title as a paragraph.
        @return: List of tokens for each paragraph.
        :param preprocess:
        """
        if preprocess:
            paras = [self.preprocess(x.get_text()) for x in soup.find_all("p")]
        else:
            paras = [[x.get_text()] for x in soup.find_all("p")]
        # If title is available add it to the paragraphs
        if title is not None:
            if preprocess:
                paras.append(self.preprocess(title))
            else:
                paras.append([title])

        return [token for para in paras for token in para]

    def get_code(
        self,
        soup: BeautifulSoup,
        get_imports_with_regex=False,
        get_functions_with_regex=False,
    ) -> (List[Import], List[Function]):
        """
        @param soup: Post body HTML wrapped in a BeautifulSoup object.
        @return: Combined string of code snippets
        """
        code_snippet = "\n".join([x.get_text() for x in soup.find_all("code")])
        try:
            syntax_tree = ast.parse(code_snippet)
        except SyntaxError:
            return ([], [])
        if get_imports_with_regex:
            modules = list(self.get_imports_via_regex(soup))
        else:
            modules = list(self.get_imports_via_ast(syntax_tree))

        if get_functions_with_regex:
            raise NotImplementedError(
                "RegEx implementation for function names not implemented yet . ."
            )
        else:
            function_defs = list(self.get_function_via_ast(syntax_tree))
        return modules, function_defs

    def to_glove_paragraph_embedding(self, tokens: List[str]) -> torch.tensor:
        """
        @param tokens: List of preprocessed tokens
        @return: Torch tensor of average word embedding
        """
        if len(tokens) == 0:
            return torch.squeeze(self._global_vectors.get_vecs_by_tokens([""]))
        word_embeddings = self._global_vectors.get_vecs_by_tokens(tokens)
        return torch.sum(word_embeddings, dim=0) / len(tokens)

    def to_bert_embedding(self, texts: List[str]) -> torch.tensor:
        encodings = self._bert_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = self._bert_model(**encodings)
            last_layer = outputs.last_hidden_state
            cls = last_layer[:, 0, :]
            return cls  # Converts from dim [1, 768] to [768]

    def to_unixcode_embedding(self, code_batches: List[str]) -> torch.tensor:
        """
        Get comments
        :param code:
        :return:
        """
        token_ids = self._unixcoder.tokenize(
            code_batches, max_length=512, mode="<encoder-only>"
        )
        longest_token_ids = max([len(x) for x in token_ids])
        token_ids = [
            x + ([self._unixcoder.config.pad_token_id] * (longest_token_ids - len(x)))
            for x in token_ids
        ]
        source_ids = torch.tensor(token_ids)
        tokens_embeddings, code_embeddings = self._unixcoder(source_ids)
        normalized_code_emb = torch.nn.functional.normalize(code_embeddings, p=2, dim=1)
        return normalized_code_emb

    def to_code_bert_embedding(self, code):
        """
        Get comments
        :param code:
        :return:
        """
        # First, get the comments from the Python code (NL)
        buf = io.StringIO(code)
        source = []
        comments = []

        token_gen = tokenize.generate_tokens(buf.readline)

        while True:
            try:
                token = next(token_gen)
                if token.type == tokenize.COMMENT:
                    comments.append(token.string)
                else:
                    source.append(token.string)
            except tokenize.TokenError:
                continue
            except StopIteration:
                break
            except IndentationError:
                continue

        nl_tokens = self._codebert_tokenizer.tokenize(" ".join(comments))

        code_tokens = self._codebert_tokenizer.tokenize(" ".join(source))

        # CodeBERT has a max token length of 512
        while len(nl_tokens) + len(code_tokens) > 509:
            if len(nl_tokens) > len(code_tokens):
                nl_tokens = nl_tokens[:-1]
            else:
                code_tokens = code_tokens[:-1]

        log.debug(f"NL Tokens: {len(nl_tokens)} Code Tokens: {len(code_tokens)}")

        tokens = (
            [self._codebert_tokenizer.cls_token]
            + nl_tokens
            + [self._codebert_tokenizer.sep_token]
            + code_tokens
            + [self._codebert_tokenizer.eos_token]
        )
        tokens_ids = self._codebert_tokenizer.convert_tokens_to_ids(tokens)

        emb = self._codebert_model(torch.tensor(tokens_ids)[None, :])[0]
        return emb.mean(dim=1).mean(dim=0)

    """
    Python RegEx methods
    """

    def get_imports_via_regex(self, soup) -> Import:
        code_snippet = "\n".join([x.get_text() for x in soup.find_all("code")])

        PATTERN = r"^\s*(?:from|import)\s+(\w+(?:\s*,\s*\w+)*)"

        for module in list(set(re.findall(PATTERN, code_snippet, flags=re.MULTILINE))):
            yield Import(module, None, None)

    """
    Python Abstract Syntax Tree methods
    """

    def get_imports_via_ast(self, syntax_tree) -> Import:
        """
        @param code_snippet:
        @return:
        """
        for node in ast.iter_child_nodes(syntax_tree):
            if isinstance(node, ast.Import):
                module = []
            elif isinstance(node, ast.ImportFrom):
                module = node.module.split(".")
            else:
                continue

            for n in node.names:
                yield Import(module, n.name.split("."), n.asname)

    def get_function_via_ast(self, syntax_tree) -> Function:
        """
        @param code_snippet:
        @return:
        """
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.FunctionDef):
                parameters = [x.arg for x in node.args.args]
                yield Function(node.name, parameters)


if __name__ == "__main__":
    pe = PostEmbedding()
    """TIME START"""
    t1 = time.time()
    for i in range(1):
        a = pe.to_unixcode_embedding(
            2 * ["\n".join(["for i in range(32):\n    #return 6 or something\n"])]
        ).shape
        b = pe.to_bert_embedding(2 * ["This is a test sentence."]).shape
    # print([x.module for x in pe.get_imports_via_regex(BeautifulSoup("<code>import ast<\code>", 'lxml'))])
    """TIME END"""
    t2 = time.time()
    print("Function=%s, Time=%s" % ("embedding", t2 - t1))
