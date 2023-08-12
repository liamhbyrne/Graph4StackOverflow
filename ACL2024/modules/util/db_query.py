"""
Database functions
"""
from datetime import datetime
from typing import List

import pandas as pd
import torch


def fetch_questions_by_post_ids(post_ids: List[int], db):
    questions_df = pd.read_sql_query(
        f"""
            SELECT * FROM Post
            WHERE PostId IN ({','.join([str(x) for x in post_ids])})
    """,
        db,
    )
    return questions_df


def fetch_questions_by_user(user_id: int, db):
    questions_df = pd.read_sql_query(
        f"""
            SELECT *
            FROM Post
            WHERE Tags LIKE '%python%' AND (PostTypeId = 1) AND ((LastEditorUserId = {user_id}) OR (OwnerUserId = {user_id}))
    """,
        db,
    )
    questions_df.set_index("PostId", inplace=True)
    return questions_df


def fetch_answers_by_user(user_id: int, db):
    answers_df = pd.read_sql_query(
        f"""
            SELECT A.Tags, B.*
            FROM Post A
                INNER JOIN Post B ON (B.ParentId = A.PostId) AND (B.ParentId IS NOT NULL)
            WHERE A.Tags LIKE '%python%' AND (B.PostTypeId = 2) AND ((B.LastEditorUserId = {user_id}) OR (B.OwnerUserId = {user_id}))
    """,
        db,
    )
    answers_df = answers_df.loc[:, ~answers_df.columns.duplicated()].copy()
    answers_df.set_index("PostId", inplace=True)
    return answers_df


def fetch_answers_for_question(question_post_id: int, db):
    """
    Fetch answers for a question for P@1 evaluation
    """
    answers_df = pd.read_sql_query(
        f"""
            SELECT *
            FROM Post
            WHERE ParentId = {question_post_id}
    """,
        db,
    )
    answers_df = answers_df.dropna(
        subset=["PostId", "Body", "Score", "OwnerUserId"]
    )
    return answers_df


def fetch_questions_by_post_ids_eval(post_ids: List[int], db):
    """
    Fetch questions for P@1 evaluation
    """
    questions_df = pd.read_sql_query(
        f"""
            SELECT * FROM Post
            WHERE PostId IN ({','.join([str(x) for x in post_ids])})
    """,
        db,
    )
    questions_df.columns = [
        "post_id",
        "question_body",
        "question_title",
        "question_user_id",
    ]
    return questions_df


def fetch_comments_by_user(user_id: int, db):
    comments_on_questions_df = pd.read_sql_query(
        f"""
            SELECT A.Tags, B.*
            FROM Post A
                INNER JOIN Comment B ON (B.PostId = A.PostId)
            WHERE A.Tags LIKE '%python%' AND (B.UserId = {user_id}) AND (A.PostTypeId = 1)
    """,
        db,
    )
    comments_on_questions_df.set_index("CommentId", inplace=True)

    comments_on_answers_df = pd.read_sql_query(
        f"""
        SELECT A.Tags, C.*
        FROM Post A
            INNER JOIN Post B ON (B.ParentId = A.PostId) AND (B.ParentId IS NOT NULL)
            INNER JOIN Comment C ON (B.PostId = C.PostId)
        WHERE A.Tags LIKE '%python%' AND (C.UserId = {user_id}) AND (B.PostTypeId = 2)
    """,
        db,
    )
    comments_on_answers_df.set_index("CommentId", inplace=True)

    return pd.concat([comments_on_questions_df, comments_on_answers_df])


def fetch_tags_for_question(question_post_id: int, db):
    tags_df = pd.read_sql_query(
        f"""
            SELECT Tags
            FROM Post
            WHERE PostId = {question_post_id}
    """,
        db,
    )
    if len(tags_df) == 0:
        return []
    return tags_df.iloc[0]["Tags"][1:-1].split("><")



