from typing import List
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from post_embedding_builder import Import, PostEmbedding
import torch_geometric.transforms as T


class StaticGraphConstruction:

    # PostEmbedding is costly to put in constructor
    post_embedding_builder = PostEmbedding()

    def __init__(self):
        self._known_tags = {}  # tag_name -> index
        self._known_modules = {}  # module_name -> index
        self._data = HeteroData()
        self._first_n_tags = 3

        self._tag_to_question_edges = []
        self._tag_to_answer_edges = []
        self._tag_to_comment_edges = []

        self._module_to_question_edges = []
        self._module_to_answer_edges = []
        self._module_to_comment_edges = []
        self._use_bert = True
        self._post_count_limit = 10



    def process_questions(self, questions: pd.DataFrame) -> torch.Tensor:
        if not len(questions):
            return None

        word_emb_batches, code_emb_batches, module_name_batches = StaticGraphConstruction.post_embedding_builder(
            questions['Body'], self._use_bert, questions['Title']
        )

        row_counter = 0
        for post_id, body, title, tags in questions[['Body', 'Title', 'Tags']].itertuples():

            modules = self.process_module_names(module_name_batches[row_counter])
            tag_list = self.parse_tag_list(tags)[:self._first_n_tags]

            for tag in tag_list:
                self._tag_to_question_edges.append((self._known_tags[tag], post_id))

            for module in modules:
                self._module_to_question_edges.append((self._known_modules[module], post_id))

            post_emb = torch.concat((word_emb_batches[row_counter], code_emb_batches[row_counter]))
            row_counter += 1
            yield post_emb


    def process_answers(self, answers: pd.DataFrame) -> torch.Tensor:
        if not len(answers):
            return None

        word_emb_batches, code_emb_batches, module_name_batches = StaticGraphConstruction.post_embedding_builder(
            answers['Body'], self._use_bert, title_batch=answers['Title']
        )

        row_counter = 0
        for i, body, title, tags in answers[['Body', 'Title', 'Tags']].itertuples():

            modules = self.process_module_names(module_name_batches[row_counter])
            tag_list = self.parse_tag_list(tags)[:self._first_n_tags]

            for tag in tag_list:
                self._tag_to_answer_edges.append((self._known_tags[tag], i))

            for module in modules:
                self._module_to_answer_edges.append((self._known_modules[module], i))

            post_emb = torch.concat((word_emb_batches[row_counter], code_emb_batches[row_counter]))
            row_counter += 1
            yield post_emb

    def process_comments(self, comments: pd.DataFrame) -> torch.Tensor:
        if not len(comments):
            return None

        word_emb_batches, code_emb_batches, module_name_batches = StaticGraphConstruction.post_embedding_builder(
            comments['Body'], self._use_bert, title_batch=[None for _ in range(len(comments))]
        )

        row_counter = 0
        for i, body, tags in comments[['Body', 'Tags']].itertuples():

            modules = self.process_module_names(module_name_batches[row_counter])
            tag_list = self.parse_tag_list(tags)[:self._first_n_tags]

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
            yield torch.rand(90)  # TODO: Map tag name to its embedding

    def process_modules(self):
        if not len(self._known_modules):
            return None
        for module in self._known_modules: # TODO: Map module name to its embedding
            yield torch.rand(110)

    """
    Utility functions
    """
    def parse_tag_list(self, tag_list: str) -> List[str]:
        tags = [x for x in tag_list[1:-1].split("><") if x not in ['python', 'python-3.x']]
        for t in tags:
            if t not in self._known_tags:
                self._known_tags[t] = len(self._known_tags)
        return tags

    def process_module_names(self, import_statements: List[Import]):
        modules = [i.module[0] for i in import_statements if i.module]
        for m in modules:
            if m not in self._known_modules:
                self._known_modules[m] = len(self._known_modules)
        return modules

    def construct(self, questions, answers, comments) -> HeteroData:
        questions = questions.head(self._post_count_limit)
        answers = answers.head(self._post_count_limit)
        comments = comments.head(self._post_count_limit)

        questions.reset_index(inplace=True)
        answers.reset_index(inplace=True)
        comments.reset_index(inplace=True)

        question_nodes = list(self.process_questions(questions))
        answer_nodes = list(self.process_answers(answers))
        comment_nodes = list(self.process_comments(comments))
        tag_nodes = list(self.process_tags())
        module_nodes = list(self.process_modules())

        # Assign node features
        self._data['question'].x = torch.stack(question_nodes) if len(question_nodes) else torch.empty(0, 1536)

        self._data['answer'].x = torch.stack(answer_nodes) if len(answer_nodes) else torch.empty(0, 1536)

        self._data['comment'].x = torch.stack(comment_nodes) if len(comment_nodes) else torch.empty(0, 768)

        self._data['tag'].x = torch.stack(tag_nodes) if len(tag_nodes) else torch.empty(0, 90)

        self._data['module'].x = torch.stack(module_nodes) if len(module_nodes) else torch.empty(0, 110)

        # Assign edge indexes
        self._data['tag', 'describes', 'question'].edge_index = torch.tensor(self._tag_to_question_edges).t().contiguous() if len(self._tag_to_question_edges) else torch.empty(2,0, dtype=torch.long)
        self._data['tag', 'describes', 'answer'].edge_index = torch.tensor(self._tag_to_answer_edges).t().contiguous() if len(self._tag_to_answer_edges) else torch.empty(2,0, dtype=torch.long)
        self._data['tag', 'describes', 'comment'].edge_index = torch.tensor(self._tag_to_comment_edges).t().contiguous() if len(self._tag_to_comment_edges) else torch.empty(2,0, dtype=torch.long)
        self._data['module', 'imported_in', 'question'].edge_index = torch.tensor(self._module_to_question_edges).t().contiguous() if len(self._module_to_question_edges) else torch.empty(2,0, dtype=torch.long)
        self._data['module', 'imported_in', 'answer'].edge_index = torch.tensor(self._module_to_answer_edges).t().contiguous() if len(self._module_to_answer_edges) else torch.empty(2,0, dtype=torch.long)

        # Remove isolated nodes, and convert to undirected graph
        graph_out = T.remove_isolated_nodes.RemoveIsolatedNodes()(self._data)
        graph_out = T.ToUndirected()(graph_out)
        graph_out.metadata()

        return graph_out