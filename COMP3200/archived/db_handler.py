import sqlite3

import pandas as pd


class DBHandler:
    def __init__(self, file_name: str):
        self._db = sqlite3.connect(file_name)

    def fetch_questions_by_user_id(self, user_id: int):
        questions = pd.read_sql_query(
            f"""
                SELECT *
                FROM Post
                WHERE Tags LIKE '%python%' AND (PostTypeId = 1) AND ((LastEditorUserId = {user_id}) OR (OwnerUserId = {user_id}))
        """,
            self._db,
        )
        questions.set_index("PostId", inplace=True)

    def fetch_answers_by_user_id(self, user_id: int):
        answers = pd.read_sql_query(
            f"""
                SELECT A.Tags, B.*
                FROM Post A
                    INNER JOIN Post B ON (B.ParentId = A.PostId) AND (B.ParentId IS NOT NULL)
                WHERE A.Tags LIKE '%python%' AND (B.PostTypeId = 2) AND ((B.LastEditorUserId = {user_id}) OR (B.OwnerUserId = {user_id}))
        """,
            self._db,
        )
        # Only use the first of the two 'Tags' columns
        answers = answers.loc[:, ~answers.columns.duplicated()].copy()
        answers.set_index("PostId", inplace=True)
        return answers

    def fetch_comments_on_questions_by_user_id(self, user_id: int):
        comments_on_questions = pd.read_sql_query(
            f"""
                SELECT A.Tags, B.*
                FROM Post A
                    INNER JOIN Comment B ON (B.PostId = A.PostId)
                WHERE A.Tags LIKE '%python%' AND (B.UserId = {user_id}) AND (A.PostTypeId = 1)
        """,
            self._db,
        )
        comments_on_questions.set_index("CommentId", inplace=True)
        return comments_on_questions

    def fetch_comments_on_answers_by_user_id(self, user_id: int):
        comments_on_answers = pd.read_sql_query(
            f"""
            SELECT A.Tags, C.*
            FROM Post A
                INNER JOIN Post B ON (B.ParentId = A.PostId) AND (B.ParentId IS NOT NULL)
                INNER JOIN Comment C ON (B.PostId = C.PostId)
            WHERE A.Tags LIKE '%python%' AND (C.UserId = {user_id}) AND (B.PostTypeId = 2)
        """,
            self._db,
        )
        comments_on_answers.set_index("CommentId", inplace=True)
        return comments_on_answers
