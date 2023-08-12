import logging
from typing import List

import pandas as pd
import torch
import torch_geometric.transforms as T
import yaml
from torch_geometric.data import HeteroData

from ACL2024.modules.embeddings.post_embedding_builder import Import, PostEmbedding
from ACL2024.modules.util.BatchedHeteroData import BatchedHeteroData

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

with open("dataset_config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)['graph_construction']


class StaticGraphConstruction:
    def __init__(
            self,
            post_embedding_builder: PostEmbedding,
            tag_embedding_model,
            module_embedding_model,
    ):
        # PostEmbedding is costly to instantiate in each StaticGraphConstruction instance.

        self._post_embedding_builder = post_embedding_builder
        self._tag_embedding_model = tag_embedding_model
        self._module_embedding_model = module_embedding_model

        self._known_tags = {}  # tag_name -> index
        self._known_modules = {}  # module_name -> index
        self._data = BatchedHeteroData()

        self._tag_to_question_edges = []
        self._tag_to_answer_edges = []
        self._tag_to_comment_edges = []

        self._module_to_question_edges = []
        self._module_to_answer_edges = []
        self._module_to_comment_edges = []

        """
        Configurable parameters
        """
        self._use_bert = CONFIG['use_bert_embeddings']
        self._graph_question_limit = CONFIG['graph_question_limit']
        self._graph_answer_limit = CONFIG['graph_answer_limit']
        self._graph_comment_limit = CONFIG['graph_comment_limit']
        self._first_n_tags = CONFIG['num_tags_per_post']

    def process_questions(self, questions: pd.DataFrame) -> torch.Tensor:
        if not len(questions):
            return None

        (
            word_emb_batches,
            code_emb_batches,
            module_name_batches,
        ) = self._post_embedding_builder(
            questions["Body"], self._use_bert, questions["Title"]
        )

        row_counter = 0
        for post_id, body, title, tags in questions[
            ["Body", "Title", "Tags"]
        ].itertuples():
            modules = self.process_module_names(module_name_batches[row_counter])
            tag_list = self.parse_tag_list(tags)[: self._first_n_tags]

            for tag in tag_list:
                self._tag_to_question_edges.append((self._known_tags[tag], post_id))

            for module in modules:
                self._module_to_question_edges.append(
                    (self._known_modules[module], post_id)
                )

            post_emb = torch.concat(
                (word_emb_batches[row_counter], code_emb_batches[row_counter])
            )
            row_counter += 1
            yield post_emb

    def process_answers(self, answers: pd.DataFrame) -> torch.Tensor:
        if not len(answers):
            return None

        (
            word_emb_batches,
            code_emb_batches,
            module_name_batches,
        ) = self._post_embedding_builder(
            answers["Body"], self._use_bert, title_batch=answers["Title"]
        )

        row_counter = 0
        for i, body, title, tags in answers[["Body", "Title", "Tags"]].itertuples():
            modules = self.process_module_names(module_name_batches[row_counter])
            tag_list = self.parse_tag_list(tags)[: self._first_n_tags]

            for tag in tag_list:
                self._tag_to_answer_edges.append((self._known_tags[tag], i))

            for module in modules:
                self._module_to_answer_edges.append((self._known_modules[module], i))

            post_emb = torch.concat(
                (word_emb_batches[row_counter], code_emb_batches[row_counter])
            )
            row_counter += 1
            yield post_emb

    def process_comments(self, comments: pd.DataFrame) -> torch.Tensor:
        if not len(comments):
            return None

        (
            word_emb_batches,
            code_emb_batches,
            module_name_batches,
        ) = self._post_embedding_builder(
            comments["Body"],
            self._use_bert,
            title_batch=[None for _ in range(len(comments))],
        )

        row_counter = 0
        for i, body, tags in comments[["Body", "Tags"]].itertuples():
            modules = self.process_module_names(module_name_batches[row_counter])
            tag_list = self.parse_tag_list(tags)[: self._first_n_tags]

            for tag in tag_list:
                self._tag_to_comment_edges.append((self._known_tags[tag], i))

            for module in modules:
                self._module_to_comment_edges.append((self._known_modules[module], i))

            post_emb = word_emb_batches[row_counter]
            row_counter += 1
            yield post_emb

    def process_tags(self):
        if not len(self._known_tags):
            return None
        for tag in self._known_tags:
            yield self._tag_embedding_model.get_tag_embedding(tag)

    def process_modules(self):
        if not len(self._known_modules):
            return None
        for module in self._known_modules:
            yield self._module_embedding_model.get_module_embedding(module)

    """
    Utility functions
    """

    def parse_tag_list(self, tag_list: str) -> List[str]:
        tags = [
            x for x in tag_list[1:-1].split("><") if x not in ["python", "python-3.x"]
        ]
        for t in tags:
            if t not in self._known_tags:
                self._known_tags[t] = len(self._known_tags)
        return tags

    def process_module_names(self, import_statements: List[Import]):
        modules = [i.module for i in import_statements if i.module]
        for m in modules:
            if m not in self._known_modules:
                self._known_modules[m] = len(self._known_modules)
        return modules

    def construct(self, questions, answers, comments) -> HeteroData:
        questions = questions.head(self._graph_question_limit)
        answers = answers.head(self._graph_question_limit)
        comments = comments.head(self._graph_comment_limit)

        questions.reset_index(inplace=True)
        answers.reset_index(inplace=True)
        comments.reset_index(inplace=True)

        question_nodes = list(self.process_questions(questions))
        answer_nodes = list(self.process_answers(answers))
        comment_nodes = list(self.process_comments(comments))
        tag_nodes = list(self.process_tags())
        module_nodes = list(self.process_modules())

        # Print node counts
        log.debug(
            len(question_nodes),
            len(answer_nodes),
            len(comment_nodes),
            len(tag_nodes),
            len(module_nodes),
            sum(
                [
                    len(question_nodes),
                    len(answer_nodes),
                    len(comment_nodes),
                    len(tag_nodes),
                    len(module_nodes),
                ]
            ),
        )

        # Print tags and modules with their indices (including offset)
        OFFSET = len(question_nodes) + len(answer_nodes) + len(comment_nodes)
        p1 = {v + (OFFSET + len(tag_nodes)): k for k, v in self._known_modules.items()}
        p2 = {v + OFFSET: k for k, v in self._known_tags.items()}
        log.debug(f"TAGS {p2} MODULES {p1}")
        # Assign node features
        self._data["question"].x = (
            torch.stack(question_nodes) if len(question_nodes) else torch.empty(0, 1536)
        )

        self._data["answer"].x = (
            torch.stack(answer_nodes) if len(answer_nodes) else torch.empty(0, 1536)
        )

        self._data["comment"].x = (
            torch.stack(comment_nodes) if len(comment_nodes) else torch.empty(0, 768)
        )

        self._data["tag"].x = (
            torch.stack(tag_nodes) if len(tag_nodes) else torch.empty(0, 50)
        )

        self._data["module"].x = (
            torch.stack(module_nodes) if len(module_nodes) else torch.empty(0, 30)
        )

        # Assign edge indexes
        self._data["tag", "describes", "question"].edge_index = (
            torch.tensor(self._tag_to_question_edges).t().contiguous()
            if len(self._tag_to_question_edges)
            else torch.empty(2, 0, dtype=torch.long)
        )
        self._data["tag", "describes", "answer"].edge_index = (
            torch.tensor(self._tag_to_answer_edges).t().contiguous()
            if len(self._tag_to_answer_edges)
            else torch.empty(2, 0, dtype=torch.long)
        )
        self._data["tag", "describes", "comment"].edge_index = (
            torch.tensor(self._tag_to_comment_edges).t().contiguous()
            if len(self._tag_to_comment_edges)
            else torch.empty(2, 0, dtype=torch.long)
        )
        self._data["module", "imported_in", "question"].edge_index = (
            torch.tensor(self._module_to_question_edges).t().contiguous()
            if len(self._module_to_question_edges)
            else torch.empty(2, 0, dtype=torch.long)
        )
        self._data["module", "imported_in", "answer"].edge_index = (
            torch.tensor(self._module_to_answer_edges).t().contiguous()
            if len(self._module_to_answer_edges)
            else torch.empty(2, 0, dtype=torch.long)
        )

        # Remove isolated nodes, and convert to undirected graph
        graph_out = T.remove_isolated_nodes.RemoveIsolatedNodes()(self._data)
        graph_out = T.ToUndirected()(graph_out)
        graph_out.metadata()

        return graph_out
