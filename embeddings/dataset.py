import logging
import os.path
import pickle
import re
import sqlite3
from typing import List, Any, Optional
import time

import pandas as pd
import torch
from bs4 import MarkupResemblesLocatorWarning
from torch_geometric.data import Dataset, download_url, Data, HeteroData
from torch_geometric.data.hetero_data import NodeOrEdgeStorage
from tqdm import tqdm
import warnings

from custom_logger import setup_custom_logger

warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

from post_embedding_builder import PostEmbedding
from static_graph_construction import StaticGraphConstruction

log = setup_custom_logger('dataset', logging.INFO)

TIMES = []

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
            return os.listdir(os.path.join(self.root, "processed"))
        return []

    def download(self):
        pass

    def get_unprocessed_ids(self):
        # Load IDs of questions to use.
        with open("../data/raw/valid_questions.pkl", 'rb') as f:
            question_ids = pickle.load(f)

        processed = []
        max_idx = -1
        for f in os.listdir(os.path.join(self.root, "processed")):
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
        # Fetch the unprocessed questions and the next index to use.
        unprocessed, idx = self.get_unprocessed_ids()


        # Fetch questions from database.
        valid_questions = self.fetch_questions_by_post_ids(unprocessed)


        for _, question in tqdm(valid_questions.iterrows(), total=len(valid_questions)):

            # Build Question embedding
            question_word_embs, question_code_embs, _ = self._post_embedding_builder(
                [question["Body"]],
                use_bert=True,
                title_batch=[question["Title"]]
            )
            question_emb = torch.concat((question_word_embs[0], question_code_embs[0]))

            # Fetch answers to question
            answers_to_question = self.fetch_answers_for_question(question["PostId"])

            # Build Answer embeddings
            for _, answer in answers_to_question.iterrows():
                label = torch.tensor([1 if answer["Score"] > 0 else 0], dtype=torch.long)
                answer_word_embs, answer_code_embs, _ = self._post_embedding_builder(
                    [answer["Body"]], use_bert=True, title_batch=[None]
                )
                answer_emb = torch.concat((answer_word_embs[0], answer_code_embs[0]))

                # Build graph
                start = time.time()
                graph: HeteroData = self.construct_graph(answer["OwnerUserId"])
                end = time.time()
                log.info(f"Graph construction took {end-start} seconds.")
                TIMES.append(end-start)
                log.info(f"Average graph construction time: {sum(TIMES)/len(TIMES)} seconds. {TIMES}")

                # pytorch geometric data object
                graph.__setattr__('question_emb', question_emb)
                graph.__setattr__('answer_emb', answer_emb)
                graph.__setattr__('score', answer["Score"])
                graph.__setattr__('question_id', question["PostId"])
                graph.__setattr__('answer_id', answer["PostId"])
                graph.__setattr__('label', label)
                torch.save(graph, os.path.join(self.processed_dir, f'data_{idx}_question_id_{question["PostId"]}_answer_id_{answer["PostId"]}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)-2

    def get(self, idx):
        file_name = [filename for filename in os.listdir(os.path.join(self.root, 'processed')) if filename.startswith(f"data_{idx}")]
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
                SELECT * FROM Post
                WHERE PostId IN ({','.join([str(x) for x in post_ids])})
        """, self._db)
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
        """
        Fetch answers for a question for P@1 evaluation
        """
        answers_df = pd.read_sql_query(f"""
                SELECT *
                FROM Post
                WHERE ParentId = {question_post_id}
        """, self._db)
        answers_df = answers_df.dropna(subset=['PostId', 'Body', 'Score', 'OwnerUserId'])
        return answers_df

    def fetch_questions_by_post_ids_eval(self, post_ids: List[int]):
        """
        Fetch questions for P@1 evaluation
        """
        questions_df = pd.read_sql_query(f"""
                SELECT * FROM Post
                WHERE PostId IN ({','.join([str(x) for x in post_ids])})
        """, self._db)
        questions_df.columns = ['post_id', 'question_body', 'question_title', 'question_user_id']
        return questions_df

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

    def fetch_tags_for_question(self, question_post_id: int):
        tags_df = pd.read_sql_query(f"""
                SELECT Tags
                FROM Post
                WHERE PostId = {question_post_id}
        """, self._db)
        if len(tags_df) == 0:
            return []
        return tags_df.iloc[0]['Tags'][1:-1].split("><")

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


    ds = UserGraphDataset('../datav2/', db_address='../stackoverflow.db', skip_processing=False)
    data = ds.get(1078)
    print("Question ndim:", data.x_dict['question'].shape)
    print("Answer ndim:", data.x_dict['answer'].shape)
    print("Comment ndim:", data.x_dict['comment'].shape)
    print("Tag ndim:", data.x_dict['tag'].shape)
    print("Module ndim:", data.x_dict['module'].shape)
    print("Question:", data.question_emb.shape)
    print("Answer:", data.answer_emb.shape)
