import ast
import time
from collections import namedtuple
from typing import List

from bs4 import BeautifulSoup
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.vocab import GloVe
from transformers import BertTokenizer, BertModel

Import = namedtuple("Import", ["module", "name", "alias"])
Function = namedtuple("Function", ["function_name", "parameter_names"])


class PostEmbedding(nn.Module):
    """
    Torch module for transforming Stackoverflow posts into a torch tensor.
    """

    def __init__(self):
        super().__init__()
        self._global_vectors = GloVe(name='840B', dim=300)
        self._en = spacy.load('en_core_web_sm')
        self._stopwords = self._en.Defaults.stop_words
        self._bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    def forward(self, html: str, title: str=None, flatten=True) -> torch.tensor:
        """
        @param html: HTML string of the body of a StackOverflow post.
        @param title: Title of a question post.
        @param flatten: Merge all paragraphs together
        @return: Post embedding represented as a torch tensor
        """

        soup = BeautifulSoup(html, 'lxml')
        ps = self.get_paragraphs(soup, title)

        para_emb = self.to_glove_paragraph_embedding(ps)

        code = self.get_code(soup)

        return para_emb, code

    def preprocess(self, text: str) -> List[str]:
        """
        @param text: Paragraph text from StackOverflow body
        @return: List of tokens which do not include stopwords, punctuation or numbers.
        """
        doc = self._en(text.lower())
        tokens = [word.text for word in doc if not (word.is_stop or word.is_punct or word.like_num)]
        return tokens

    def get_paragraphs(self, soup: BeautifulSoup, title: str = None) -> List[str]:
        """
        @param soup: Post body HTML wrapped in a BeautifulSoup object.
        @param title: If available, add title as a paragraph.
        @return: List of tokens for each paragraph.
        """
        paras = [self.preprocess(x.get_text()) for x in soup.find_all('p')]

        # If title is available add it to the paragraphs
        if title is not None:
            paras.append(self.preprocess(title))
        return [token for para in paras for token in para]

    def get_code(self, soup: BeautifulSoup) -> (List[Import], List[Function]):
        """
        @param soup: Post body HTML wrapped in a BeautifulSoup object.
        @return: Combined string of code snippets
        """
        code_snippet = "\n".join([x.get_text() for x in soup.find_all('code')])
        try:
            syntax_tree = ast.parse(code_snippet)
        except SyntaxError:
            return ([],[])
        modules = list(self.get_imports(syntax_tree))
        function_defs = list(self.get_function(syntax_tree))
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

    def to_bert_embedding(self, text: str) -> torch.tensor:
        sentences = [i.text for i in self._en(text).sents]
        encodings = self._tokenizer(sentences, padding=True, return_tensors='pt')
        with torch.no_grad():
            embeds = self._bert_model(**encodings)
        return embeds.mean(dim=1).mean(dim=0)

    """
    Python Abstract Syntax Tree methods
    """

    def get_imports(self, syntax_tree) -> Import:
        """
        @param code_snippet:
        @return:
        """
        for node in ast.iter_child_nodes(syntax_tree):
            if isinstance(node, ast.Import):
                module = []
            elif isinstance(node, ast.ImportFrom):
                module = node.module.split('.')
            else:
                continue

            for n in node.names:
                yield Import(module, n.name.split('.'), n.asname)

    def get_function(self, syntax_tree) -> Function:
        """
        @param code_snippet:
        @return:
        """
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.FunctionDef):
                parameters = [x.arg for x in node.args.args]
                yield Function(node.name, parameters)


if __name__ == '__main__':
    pe = PostEmbedding()
