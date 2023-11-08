import logging
import os
import pickle
import sqlite3
import warnings

import pandas as pd
import yaml
from bs4 import MarkupResemblesLocatorWarning, BeautifulSoup
from tqdm import tqdm

from ACL2024.modules.util.custom_logger import setup_custom_logger
from ACL2024.modules.util.db_query import (
    fetch_questions_by_post_ids,
    fetch_answers_for_question,
)
from ACL2024.modules.util.get_root_dir import get_project_root

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

log = setup_custom_logger("dataset", logging.INFO)

with open(
    os.path.join(get_project_root(), "modules", "dataset", "dataset_config.yaml"), "r"
) as file:
    CONFIG = yaml.safe_load(file)["prompt_response_dataset"]


class PromptResponseDataset:
    def __init__(self, valid_questions_pkl_path=None, db_address: str = None):
        self._valid_questions_pkl_path = os.path.join(
            get_project_root(), valid_questions_pkl_path
        )

        self._db_address = os.path.join(get_project_root(), db_address)
        self._db = sqlite3.connect(self._db_address)

        with open(self._valid_questions_pkl_path, "rb") as f:
            self._question_ids = pickle.load(f)[:100]

    def process(self):
        """ """

        # Fetch questions from database.
        valid_questions = fetch_questions_by_post_ids(self._question_ids, self._db)

        output = []

        for _, question in tqdm(valid_questions.iterrows(), total=len(valid_questions)):
            log.debug(f"Processing question {question['PostId']}")

            # Fetch content from answers
            answers_to_question = fetch_answers_for_question(
                question["PostId"], self._db
            )
            answers_to_question["Body"] = answers_to_question["Body"].apply(
                self.get_raw_text_from_post
            )

            # Identify accepted answer
            accepted_answer = answers_to_question[
                answers_to_question["PostId"] == question["AcceptedAnswerId"]
            ]

            if CONFIG["create_question_answer_pairs"]:
                # Create prompt-response pairs
                prompt_response_pairs = self.create_question_answer_pairs(
                    question, answers_to_question
                )

            elif CONFIG["create_question_thread_pairs"]:
                # Create prompt-response pairs
                prompt_response_pairs = self.create_question_thread_pairs(
                    question, answers_to_question
                )
            else:
                raise ValueError("No valid dataset type specified.")

            output.append(prompt_response_pairs)

        self._output = output

    def get_raw_text_from_post(self, html: str):
        """ """
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text()

    def create_question_answer_pairs(self, question, answers_to_question: pd.DataFrame):
        """ """

        # Create prompt-response pairs

        question_content = self.get_raw_text_from_post(question["Body"])

        prompt_response_pairs = []
        for _, answer in answers_to_question.iterrows():
            prompt_response_pairs.append(
                {
                    "STACKOVERFLOW_QUESTION:": question_content,
                    "STACKOVERFLOW_RESPONSE": answer["Body"],
                    "label": 1
                    if answer["PostId"] == question["AcceptedAnswerId"]
                    else 0,
                }
            )

        return prompt_response_pairs

    def create_question_thread_pairs(self, question, answers_to_question: pd.DataFrame):
        """ """
        question_content = self.get_raw_text_from_post(question["Body"])

        # Response string should be in the form: RESPONSE 1 <content> RESPONSE 2 <content> ... RESPONSE N <content>
        response_string = " ".join(
            [
                f"RESPONSE {i} {self.get_raw_text_from_post(answer['Body'])}"
                for i, answer in answers_to_question.iterrows()
            ]
        )

        if question["AcceptedAnswerId"] in answers_to_question["PostId"].values:
            accepted_index = answers_to_question[
                answers_to_question["PostId"] == question["AcceptedAnswerId"]
            ].index[0]
        else:
            accepted_index = "None Accepted"

        return {
            "STACKOVERFLOW_QUESTION:": question_content,
            "STACKOVERFLOW_RESPONSE": response_string,
            "accepted_index": accepted_index,
        }

    def write_to_csv(self, output_path):
        """ """
        df = pd.DataFrame(self._output)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    """ """

    ds = PromptResponseDataset(
        db_address=CONFIG["db_address"],
        valid_questions_pkl_path=CONFIG["valid_questions_pkl_path"],
    )
    ds.process()
    ds.write_to_csv("test.csv")

    print(pd.read_csv("test.csv"))
