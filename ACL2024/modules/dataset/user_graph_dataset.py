import gc
import logging
import os.path
import pickle
import re
import sqlite3
import time
import warnings
from datetime import datetime

import pandas as pd
import torch
import yaml
from bs4 import MarkupResemblesLocatorWarning
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

from ACL2024.modules.dataset.static_graph_construction import StaticGraphConstruction
from ACL2024.modules.embeddings.module_embedding import ModuleEmbeddingTrainer
from ACL2024.modules.embeddings.post_embedding_builder import PostEmbedding
from ACL2024.modules.embeddings.tag_embedding import NextTagEmbeddingTrainer
from ACL2024.modules.util.custom_logger import setup_custom_logger
from ACL2024.modules.util.db_query import fetch_questions_by_post_ids, fetch_tags_for_question, fetch_answers_for_question, fetch_questions_by_user, fetch_answers_by_user, fetch_comments_by_user
from ACL2024.modules.util.get_root_dir import get_project_root

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

log = setup_custom_logger("dataset", logging.INFO)

with open("dataset_config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)['user_graph_dataset']


class UserGraphDataset(Dataset):
    tag_embedding_model = NextTagEmbeddingTrainer.load_model(
        **CONFIG['tag_embeddings']
    )
    module_embedding_model = ModuleEmbeddingTrainer.load_model(
        **CONFIG['module_embeddings']
    )

    def __init__(
            self,
            root,
            valid_questions_pkl_path=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            db_address: str = None,
            skip_processing=False,
    ):
        self._skip_processing = skip_processing
        self._valid_questions_pkl_path = os.path.join(get_project_root(), valid_questions_pkl_path)

        self._db_address = os.path.join(get_project_root(), db_address)
        self._post_embedding_builder = PostEmbedding()
        self._db = sqlite3.connect(self._db_address)
        # Call init last, as it may trigger the process function.
        super().__init__(os.path.join(get_project_root(), root), transform, pre_transform, pre_filter)

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
        with open(self._valid_questions_pkl_path, "rb") as f:
            question_ids = pickle.load(f)

        log.info(f"Found {len(question_ids)} valid questions.")

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
        return unprocessed, max_idx + 1

    def process(self):
        """ """

        # Fetch the unprocessed questions and the next index to use.
        unprocessed, idx = self.get_unprocessed_ids()

        # Fetch questions from database.
        valid_questions = fetch_questions_by_post_ids(unprocessed, self._db)

        for _, question in tqdm(valid_questions.iterrows(), total=len(valid_questions)):
            log.debug(f"Processing question {question['PostId']}")

            # Build Question embedding
            question_word_embs, question_code_embs, _ = self._post_embedding_builder(
                [question["Body"]], use_bert=True, title_batch=[question["Title"]]
            )
            question_emb = torch.concat(
                (question_word_embs[0], question_code_embs[0])
            ).detach()
            question_metadata = self.fetch_question_metadata(
                question["PostId"], self._db
            )

            # Fetch answers to question
            answers_to_question = fetch_answers_for_question(
                question["PostId"], self._db
            )

            for _, answer in answers_to_question.iterrows():
                # Create instance
                self.create_instance(
                    answer, question, idx, question_emb, question_metadata
                )

                gc.collect()
                torch.cuda.empty_cache()

                # Increment index
                idx += 1

    def create_instance(self, answer, question, idx, question_emb, question_metadata):
        label = torch.tensor([1 if answer["Score"] > 0 else 0], dtype=torch.long)
        answer_word_embs, answer_code_embs, _ = self._post_embedding_builder(
            [answer["Body"]], use_bert=True, title_batch=[None]
        )
        answer_emb = torch.concat((answer_word_embs[0], answer_code_embs[0]))

        answer_metadata = self.fetch_answer_metadata(answer["PostId"], self._db)

        # Build graph
        start = time.time()
        graph: HeteroData = self.construct_graph(
            answer["OwnerUserId"], sqlite3.connect(self._db_address)
        )
        end = time.time()
        log.debug(f"Graph construction took {end - start} seconds.")

        # Add attributes based on config
        if CONFIG['add_question_embedding']:
            graph.__setattr__("question_emb", question_emb)

        if CONFIG['add_question_metadata']:
            graph.__setattr__("question_metadata", question_metadata)

        if CONFIG['add_answer_embedding']:
            graph.__setattr__("answer_emb", answer_emb)

        if CONFIG['add_answer_metadata']:
            graph.__setattr__("answer_metadata", answer_metadata)

        if CONFIG['add_score']:
            graph.__setattr__("score", answer["Score"])

        if CONFIG['add_question_id']:
            graph.__setattr__("question_id", question["PostId"])

        if CONFIG['add_answer_id']:
            graph.__setattr__("answer_id", answer["PostId"])

        if CONFIG['add_user_id']:
            graph.__setattr__("user_id", answer["OwnerUserId"])

        if CONFIG['add_label']:
            graph.__setattr__("label", label)

        if CONFIG['add_accepted']:
            graph.__setattr__("accepted", answer["AcceptedAnswerId"] == answer["PostId"])

        if CONFIG['add_user_info']:
            user_info = self.fetch_user_info(answer["OwnerUserId"], self._db)
            graph.__setattr__("user_info", user_info)

        torch.save(
            graph,
            os.path.join(
                self.processed_dir,
                f'data_{idx}_question_id_{question["PostId"]}_answer_id_{answer["PostId"]}.pt',
            ),
        )
        log.debug(
            f"Saved data_{idx}_question_id_{question['PostId']}_answer_id_{answer['PostId']}.pt"
        )

    def len(self):
        return len(self.processed_file_names) - 2

    def get(self, idx):
        file_name = [
            filename
            for filename in os.listdir(os.path.join(self.root, "processed"))
            if filename.startswith(f"data_{idx}")
        ]
        if len(file_name):
            data = torch.load(os.path.join(self.processed_dir, file_name[0]))

            return data
        else:
            raise Exception(f"Data with index {idx} not found.")

    def fetch_question_metadata(self, question_post_id: int, db) -> torch.tensor:
        """
        Builds a vector containing:
        1. View count
        2. Creation Date
        3. Recent Activity Date
        4. Tag Embeddings
        5. Number of comments
        """

        view_count, creation_date, last_edit_date = pd.read_sql_query(
            f"""
                SELECT ViewCount, CreationDate, LastEditDate
                FROM Post
                WHERE PostId = {question_post_id}
        """,
            db,
        ).iloc[0]

        if last_edit_date is None:
            last_edit_date = creation_date

        # Encode dates as a min-max value
        last_edit_date = (
                                 datetime.strptime(last_edit_date, "%Y-%m-%dT%H:%M:%S.%f")
                                 - datetime(2008, 7, 31)
                         ).days / (datetime(2023, 12, 31) - datetime(2008, 7, 31)).days
        creation_date = (
                                datetime.strptime(creation_date, "%Y-%m-%dT%H:%M:%S.%f")
                                - datetime(2008, 7, 31)
                        ).days / (datetime(2023, 12, 31) - datetime(2008, 7, 31)).days

        # Encode view count as a min-max value
        view_count = view_count / 1000000

        tags = fetch_tags_for_question(question_post_id, db)[:5]
        tag_embeddings = torch.zeros(len(tags), 50)
        for i, tag in enumerate(tags):
            tag_embeddings[i] = self.tag_embedding_model.get_tag_embedding(tag)

        comments_count = pd.read_sql_query(
            f"""
                SELECT COUNT(*)
                FROM Comment
                WHERE PostId = {question_post_id}
        """,
            db,
        ).iloc[0]["COUNT(*)"]

        question_metadata = {
            "view_count": view_count,
            "creation_date": creation_date,
            "last_edit_date": last_edit_date,
            "comments_count": comments_count,
            "tag_embeddings": tag_embeddings,
        }

        return question_metadata

    def fetch_answer_metadata(self, answer_post_id: int, db) -> torch.tensor:
        """
        Builds a vector containing:
        1. View count
        2. Creation Date
        3. Recent Activity Date
        4. Number of comments
        """
        creation_date = pd.read_sql_query(
            f"""
                SELECT CreationDate
                FROM Post
                WHERE PostId = {answer_post_id}
        """,
            db,
        ).iloc[0]["CreationDate"]
        # Encode date as a min-max value
        creation_date = (
                                datetime.strptime(creation_date, "%Y-%m-%dT%H:%M:%S.%f")
                                - datetime(2008, 7, 31)
                        ).days / (datetime(2023, 12, 31) - datetime(2008, 7, 31)).days

        comments_count = pd.read_sql_query(
            f"""
                SELECT COUNT(*)
                FROM Comment
                WHERE PostId = {answer_post_id}
        """,
            db,
        ).iloc[0]["COUNT(*)"]

        answer_metadata = {
            "creation_date": creation_date,
            "comments_count": comments_count,
        }

        return answer_metadata

    def fetch_user_info(self, user_id: int, db) -> torch.tensor:
        """
        Builds a vector containing:
        1. User Creation Date
        2. Reputation
        3. Number of answers
        4. Number of comments
        5. Number of accepted answers
        6. Badge Vector
        7. Top-n tag embeddings
        """

        user_creation_date, reputation = pd.read_sql_query(
            f"""
                SELECT CreationDate, Reputation
                FROM User
                WHERE UserId = {user_id}
        """,
            db,
        ).iloc[0]
        # Encode date as a min-max value
        user_creation_date = (
                                     datetime.strptime(user_creation_date, "%Y-%m-%dT%H:%M:%S.%f")
                                     - datetime(2008, 7, 31)
                             ).days / (datetime(2023, 12, 31) - datetime(2008, 7, 31)).days

        # Encode reputation using MinMax
        reputation = reputation / 200000

        answers_count = pd.read_sql_query(
            f"""
                SELECT COUNT(*)
                FROM Post
                WHERE OwnerUserId = {user_id} AND PostTypeId = 2
        """,
            db,
        ).iloc[0]["COUNT(*)"]
        # Encode answers count using MinMax
        answers_count = answers_count / 1000

        comments_count = pd.read_sql_query(
            f"""
                SELECT COUNT(*)
                FROM Comment
                WHERE UserId = {user_id}
        """,
            db,
        ).iloc[0]["COUNT(*)"]
        # Encode comments count using MinMax
        comments_count = comments_count / 1000

        accepted_answers_count = pd.read_sql_query(
            f"""
                SELECT COUNT(*) FROM Post A
                JOIN POST Q ON A.ParentId = Q.PostId
                WHERE A.OwnerUserId = {user_id} AND A.PostId = Q.AcceptedAnswerId
        """,
            db,
        ).iloc[0]["COUNT(*)"]
        # Encode accepted answers count using MinMax
        accepted_answers_count = accepted_answers_count / 10000

        # Count bronze, silver and gold badges
        badges = pd.read_sql_query(
            f"""
                SELECT Class
                FROM Badge
                WHERE UserId = {user_id}
        """,
            db,
        )
        badge_vector = torch.zeros(3)
        for badge in badges["Class"]:
            if badge == "1":
                badge_vector[0] += 1
            elif badge == "2":
                badge_vector[1] += 1
            elif badge == "3":
                badge_vector[2] += 1

        top_n_tags = pd.read_sql_query(
            f"""
                SELECT Tags
                FROM Post
                WHERE OwnerUserId = {user_id} AND PostTypeId = 1
        """,
            db,
        )
        top_n_tags = (
            top_n_tags["Tags"]
            .str[1:-1]
            .str.split("><")
            .explode()
            .value_counts()[:CONFIG['user_info_top_n_tags']]
            .index.tolist()
        )

        top_n_tag_embs = torch.zeros(len(top_n_tags), 50)
        for i, tag in enumerate(top_n_tags):
            top_n_tag_embs[i] = self.tag_embedding_model.get_tag_embedding(tag)

        user_info = {
            "user_creation_date": user_creation_date,
            "reputation": reputation,
            "answers_count": answers_count,
            "comments_count": comments_count,
            "accepted_answers_count": accepted_answers_count,
            "badge_vector": badge_vector,
            "top_n_tags": top_n_tag_embs,
        }

        return user_info
    def construct_graph(self, user_id: int, db):
        graph_constructor = StaticGraphConstruction(
            post_embedding_builder=self._post_embedding_builder,
            tag_embedding_model=self.tag_embedding_model,
            module_embedding_model=self.module_embedding_model,
        )
        qs = fetch_questions_by_user(user_id, db)
        ans = fetch_answers_by_user(user_id, db)
        cs = fetch_comments_by_user(user_id, db)
        return graph_constructor.construct(questions=qs, answers=ans, comments=cs)


if __name__ == "__main__":
    """
    Build List of question post_ids.
    This will be fed into the Dataset class to construct the graphs.
    This setup allows us to have a fixed set of questions/answers
    for each dataset (rather than selecting new questions each time).
    """

    ds = UserGraphDataset(
        root=CONFIG['root'],
        db_address=CONFIG['db_address'],
        skip_processing=CONFIG['skip_processing'],
        valid_questions_pkl_path=CONFIG['valid_questions_pkl_path']
    )
