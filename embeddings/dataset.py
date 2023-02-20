import logging
import os.path
import pickle
import re
import sqlite3
from typing import List
import time

import pandas as pd
import torch
from bs4 import MarkupResemblesLocatorWarning
from torch_geometric.data import Dataset, download_url, Data, HeteroData
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

from post_embedding_builder import PostEmbedding
from static_graph_construction import StaticGraphConstruction

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger("dataset")


class UserGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, db_address:str=None, skip_processing=False):
        self._skip_processing = skip_processing

        # Connect to database.
        if db_address is not None:
            self._db = sqlite3.connect(db_address)
            self._post_embedding_builder = PostEmbedding()
        # Call init last, as it may trigger the process function.
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self._skip_processing:
            return os.listdir("../data/processed")
        return []

    def download(self):
        pass

    def get_unprocessed_ids(self):
        # Load IDs of questions to use.
        with open("../data/raw/valid_questions.pkl", 'rb') as f:
            question_ids = pickle.load(f)

        processed = []
        max_idx = -1
        for f in os.listdir("../data/processed"):
            question_id_search = re.search(r"id_(\d+)", f)
            if question_id_search:
                processed.append(int(question_id_search.group(1)))

            idx_search = re.search(r"data_(\d+)", f)
            if idx_search:
                next_idx = int(idx_search.group(1))
                if next_idx > max_idx:
                    max_idx = int(idx_search.group(1))

        # Fetch question ids that have not been processed yet.
        unprocessed = [q_id for q_id in question_ids if q_id not in processed]
        return unprocessed, max_idx+1

    def process(self):
        """
        """
        log.info("Processing data...")
        '''TIME START'''
        t1 = time.time()

        # Fetch the unprocessed questions and the next index to use.
        unprocessed, idx = self.get_unprocessed_ids()

        '''TIME END'''
        t2 = time.time()
        log.debug("Function=%s, Time=%s" % (self.get_unprocessed_ids.__name__, t2 - t1))

        '''TIME START'''
        t1 = time.time()

        # Fetch questions from database.
        valid_questions = self.fetch_questions_by_post_ids(unprocessed)

        '''TIME END'''
        t2 = time.time()
        log.debug("Function=%s, Time=%s" % (self.fetch_questions_by_post_ids.__name__, t2 - t1))


        for row in tqdm(valid_questions.itertuples(), total=len(valid_questions)):
            '''TIME START'''
            t1 = time.time()

            # Build Question embedding
            question_word_embs, question_code_embs, _ = self._post_embedding_builder(
                [row.question_body],
                use_bert=True,
                title_batch=[row.question_title]
            )
            question_emb = torch.concat((question_word_embs[0], question_code_embs[0]))
            '''TIME END'''
            t2 = time.time()
            log.debug("Function=%s, Time=%s" % ("Post embedding builder (question)", t2 - t1))


            '''TIME START'''
            t1 = time.time()
            # Fetch answers to question
            answers_to_question = self.fetch_answers_for_question(row.post_id)
            '''TIME END'''
            t2 = time.time()
            log.debug("Function=%s, Time=%s" % (self.fetch_answers_for_question.__name__, t2 - t1))

            # Build Answer embeddings
            for _, answer_body, answer_user_id, score in answers_to_question.itertuples():
                label = torch.tensor([1 if score > 0 else 0], dtype=torch.long)
                answer_word_embs, answer_code_embs, _ = self._post_embedding_builder(
                    [answer_body], use_bert=True, title_batch=[None]
                )
                answer_emb = torch.concat((answer_word_embs[0], answer_code_embs[0]))


                '''TIME START'''
                t1 = time.time()
                # Build graph
                graph: HeteroData = self.construct_graph(answer_user_id)
                '''TIME END'''
                t2 = time.time()
                log.debug("Function=%s, Time=%s" % (self.construct_graph.__name__, t2 - t1))

                # pytorch geometric data object
                graph.__setattr__('question_emb', question_emb)
                graph.__setattr__('answer_emb', answer_emb)
                graph.__setattr__('label', label)
                torch.save(graph, os.path.join(self.processed_dir, f'data_{idx}_question_id_{row.post_id}'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)-2

    def get(self, idx):
        file_name = [filename for filename in os.listdir('../data/processed/') if filename.startswith(f"data_{idx}")]
        if len(file_name):
            data = torch.load(os.path.join(self.processed_dir, file_name[0]))
            return data
        else:
            raise Exception(f"Data with index {idx} not found.")

    '''
    Database functions
    '''

    def fetch_questions_by_post_ids(self, post_ids: List[int]):
        questions_df = pd.read_sql_query(f"""
                SELECT PostId, Body, Title, OwnerUserId FROM Post
                WHERE PostId IN ({','.join([str(x) for x in post_ids])})
        """, self._db)
        questions_df.columns = ['post_id', 'question_body', 'question_title', 'question_user_id']
        return questions_df

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
    '''
    Build List of question post_ids.
    This will be fed into the Dataset class to construct the graphs.
    This setup allows us to have a fixed set of questions/answers
    for each dataset (rather than selecting new questions each time).
    '''


    ds = UserGraphDataset('../data/', db_address='../stackoverflow.db', skip_processing=True)
    data = ds.get(1078)
    print("Question ndim:", data.x_dict['question'].shape)
    print("Answer ndim:", data.x_dict['answer'].shape)
    print("Comment ndim:", data.x_dict['comment'].shape)
    print("Tag ndim:", data.x_dict['tag'].shape)
    print("Module ndim:", data.x_dict['module'].shape)
