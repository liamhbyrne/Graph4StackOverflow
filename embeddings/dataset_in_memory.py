import logging
import os

import torch
from torch_geometric.data import InMemoryDataset

from custom_logger import setup_custom_logger

log = setup_custom_logger('in-memory-dataset', logging.INFO)

class UserGraphDatasetInMemory(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        return ['in-memory-dataset.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        for f in self.raw_file_names:
            data = torch.load(os.path.join(self.raw_dir, f))
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[0]))



if __name__ == '__main__':
    dataset = UserGraphDatasetInMemory('../data/')

    print(dataset.get(3))