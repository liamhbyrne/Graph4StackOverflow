import logging
import os
import re
from typing import List

import torch
from torch_geometric.data import InMemoryDataset

from custom_logger import setup_custom_logger

log = setup_custom_logger('in-memory-dataset', logging.INFO)

class UserGraphDatasetInMemory(InMemoryDataset):
    def __init__(self, root, file_name_out: str, question_ids:List[int]=None, transform=None, pre_transform=None, pre_filter=None):
        self._file_name_out = file_name_out
        self._question_ids = question_ids
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data = self.data.apply(lambda x: x.detach())

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_in_memory')

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return [x for x in os.listdir("../data/processed") if x not in ['pre_filter.pt', 'pre_transform.pt']]

    @property
    def processed_file_names(self):
        return [self._file_name_out]

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        for f in self.raw_file_names:
            question_id_search = re.search(r"id_(\d+)", f)
            if question_id_search:
                if int(question_id_search.group(1)) not in self._question_ids:
                    continue

            data = torch.load(os.path.join(self.raw_dir, f))
            data_list.append(data)

        data, slices = self.collate(data_list)
        self.processed_paths[0] = f"{len(data_list)}-{self.processed_file_names[0]}"
        torch.save((data, slices), os.path.join(self.processed_paths[0]))



if __name__ == '__main__':
    question_ids = set()
    # Split by question ids
    for f in os.listdir("../data/processed"):
        question_id_search = re.search(r"id_(\d+)", f)
        if question_id_search:
            question_ids.add(int(question_id_search.group(1)))

    #question_ids = list(question_ids)[:len(question_ids)* 0.6]
    train_ids = list(question_ids)[:int(len(question_ids) * 0.7)]
    test_ids = [x for x in question_ids if x not in train_ids]

    log.info(f"Training question count {len(train_ids)}")
    log.info(f"Testing question count {len(test_ids)}")

    train_dataset = UserGraphDatasetInMemory('../data/', train_ids, f'train-{len(train_ids)}-qs.pt')
    test_dataset = UserGraphDatasetInMemory('../data/', test_ids, f'test-{len(test_ids)}-qs.pt')
