import logging
import os
import random
import re
from typing import List

import torch
from torch_geometric.data import InMemoryDataset

from ACL2024.modules.util.custom_logger import setup_custom_logger

log = setup_custom_logger("compile-dataset", logging.INFO)


class UserGraphDatasetInMemory(InMemoryDataset):
    def __init__(
        self,
        root,
        file_name_out: str,
        question_ids: List[int],
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self._file_name_out = file_name_out
        self._question_ids = question_ids
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # Remove gradient requirements
        self.data = self.data.apply(lambda x: x.detach())

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed_in_memory")

    @property
    def raw_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        return [
            x
            for x in os.listdir(os.path.join(self.root, "processed"))
            if x not in ["pre_filter.pt", "pre_transform.pt"]
        ]

    @property
    def processed_file_names(self):
        return [self._file_name_out]

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        for f in self.raw_file_names:
            log.info(f"Processing {f}")
            question_id_search = re.search(r"id_(\d+)", f)
            if question_id_search:
                if int(question_id_search.group(1)) not in self._question_ids:
                    continue

            data = torch.load(os.path.join(self.raw_dir, f))
            data_list.append(data)

        data, slices = self.collate(data_list)
        self.processed_paths[0] = f"{len(data_list)}-{self.processed_file_names[0]}"
        torch.save((data, slices), os.path.join(self.processed_paths[0]))


"""
Utility functions
"""


def fetch_question_ids(root) -> List[int]:
    question_ids = set()
    # Split by question ids
    for f in os.listdir(os.path.join(root, "processed")):
        question_id_search = re.search(r"id_(\d+)", f)
        if question_id_search:
            question_ids.add(int(question_id_search.group(1)))
    return list(question_ids)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


"""
Dataset creation functions
"""


def create_datasets_for_kfolds(folds, root):
    question_ids = fetch_question_ids(root)
    random.shuffle(question_ids)

    folds = list(split(question_ids, folds))
    for i, fold_question_ids in enumerate(folds):
        yield UserGraphDatasetInMemory(
            "../data/", f"fold-{i + 1}-{len(question_ids)}-qs.pt", fold_question_ids
        )


def create_train_test_datasets():
    question_ids = fetch_question_ids(ROOT)

    train_ids = list(question_ids)[: int(len(question_ids) * 0.7)]
    test_ids = [x for x in question_ids if x not in train_ids]

    log.info(f"Training question count {len(train_ids)}")
    log.info(f"Testing question count {len(test_ids)}")

    train_dataset = UserGraphDatasetInMemory(
        ROOT, f"train-{len(train_ids)}-qs.pt", train_ids
    )
    test_dataset = UserGraphDatasetInMemory(
        ROOT, f"test-{len(test_ids)}-qs.pt", test_ids
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    ROOT = "../../data/"
    choice = input("1. Create train/test datasets\n2. Create k-fold datasets\n>>>")
    if choice == "1":
        create_train_test_datasets()
    elif choice == "2":
        n = int(input("Enter number of folds: "))
        folds = list(create_datasets_for_kfolds(n, ROOT))
