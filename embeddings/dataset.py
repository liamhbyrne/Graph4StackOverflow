import logging
import os.path
import pickle
import sqlite3

import pandas as pd
import torch
from bs4 import MarkupResemblesLocatorWarning
from torch_geometric.data import Dataset, download_url, Data
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

from post_embedding_builder import PostEmbedding
from static_graph_construction import StaticGraphConstruction

logging.basicConfig()
#logging.getLogger().setLevel(logging.ERROR)
log = logging.getLogger("dataset")


class UserGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, db_address:str=None, question_count=70000):
        self._question_count = question_count
        # Connect to database.
        if db_address is not None:
            self._db = sqlite3.connect(db_address)
            self._post_embedding_builder = PostEmbedding()
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return os.listdir("data/processed")

    def download(self):
        pass

    def process(self):
        idx = 0
        valid_questions = self.fetch_valid_questions()
        for row in tqdm(valid_questions.itertuples(), total=len(valid_questions)):
            # Build Question embedding
            question_emb = self._post_embedding_builder(
                row.question_body,
                use_bert=True,
                title=row.question_title
            )
            answers_to_question = self.fetch_answers_for_question(row.post_id)
            # Build Answer embeddings
            for _, answer_body, answer_user_id, score in answers_to_question.itertuples():
                answer_emb = self._post_embedding_builder(
                    answer_body,
                    use_bert=True
                )
                # Build graph
                graph = self.construct_graph(answer_user_id)
                # pytorch geometric data object
                data = Data(
                    x=graph.x_dict,
                    edge_index=graph.edge_index_dict,
                    y=torch.LongTensor(1 if score > 0 else 0),
                    question_emb=question_emb,
                    answer_emb=answer_emb
                )
                torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    '''
    Database functions
    '''

    def fetch_valid_questions(self):
        valid_questions = pd.read_sql_query(f"""
                SELECT Q.PostId, Q.Body, Q.Title, Q.OwnerUserId FROM Post Q
                INNER JOIN Post A ON Q.PostId = A.ParentId
                WHERE (Q.Tags LIKE '%<python>%')
                GROUP BY A.ParentId
                HAVING SUM(A.Score) > 15
                LIMIT {self._question_count}
        """, self._db)
        valid_questions.columns = ['post_id', 'question_body', 'question_title', 'question_user_id']
        return valid_questions

    def fetch_questions_by_user(self, user_id: int):
        questions_df = pd.read_sql_query(f"""
                SELECT *
                FROM Post
                WHERE Tags LIKE '%python%' AND (PostTypeId = 1) AND ((LastEditorUserId = {user_id}) OR (OwnerUserId = {user_id}))
        """, self._db)
        questions_df.set_index('PostId', inplace=True)
        return questions_df

    def fetch_answers_by_user(self, user_id: int):
        answers_df = pd.read_sql_query(f"""
                SELECT A.Tags, B.*
                FROM Post A
                    INNER JOIN Post B ON (B.ParentId = A.PostId) AND (B.ParentId IS NOT NULL)
                WHERE A.Tags LIKE '%python%' AND (B.PostTypeId = 2) AND ((B.LastEditorUserId = {user_id}) OR (B.OwnerUserId = {user_id}))
        """, self._db)
        answers_df = answers_df.loc[:, ~answers_df.columns.duplicated()].copy()
        answers_df.set_index('PostId', inplace=True)
        return answers_df

    def fetch_answers_for_question(self, question_post_id: int):
        answers_df = pd.read_sql_query(f"""
                SELECT Body, OwnerUserId, Score
                FROM Post
                WHERE ParentId = {question_post_id}
        """, self._db)
        answers_df = answers_df.dropna()
        return answers_df

    def fetch_comments_by_user(self, user_id: int):
        comments_on_questions_df = pd.read_sql_query(f"""
                SELECT A.Tags, B.*
                FROM Post A
                    INNER JOIN Comment B ON (B.PostId = A.PostId)
                WHERE A.Tags LIKE '%python%' AND (B.UserId = {user_id}) AND (A.PostTypeId = 1)
        """, self._db)
        comments_on_questions_df.set_index('CommentId', inplace=True)

        comments_on_answers_df = pd.read_sql_query(f"""
            SELECT A.Tags, C.*
            FROM Post A
                INNER JOIN Post B ON (B.ParentId = A.PostId) AND (B.ParentId IS NOT NULL)
                INNER JOIN Comment C ON (B.PostId = C.PostId)
            WHERE A.Tags LIKE '%python%' AND (C.UserId = {user_id}) AND (B.PostTypeId = 2)
        """, self._db)
        comments_on_answers_df.set_index('CommentId', inplace=True)

        return pd.concat([comments_on_questions_df, comments_on_answers_df])

    def construct_graph(self, user_id: int):
        graph_constructor = StaticGraphConstruction()
        qs = self.fetch_questions_by_user(user_id)
        ans = self.fetch_answers_by_user(user_id)
        cs = self.fetch_comments_by_user(user_id)
        return graph_constructor.construct(questions=qs, answers=ans, comments=cs)


if __name__ == '__main__':
    ds = UserGraphDataset('../data/', db_address='../stackoverflow.db', question_count=100)
    print(ds.get(0))