import itertools
import logging
import random
import sqlite3
from typing import *

import pandas as pd
import torch
from bs4 import BeautifulSoup
from torch import nn, optim
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from post_embedding_builder import PostEmbedding

logging.basicConfig(level=logging.INFO)


class ModuleEmbeddingTrainer:

    def __init__(self, emb_size: int, database_path: str = None):
        logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Proceeding with {self.device} . .")
        if database_path is not None:
            self.db = sqlite3.connect(database_path)
            logger.info(f"Connected to {database_path}")
        self.emb_size = emb_size
        self.emb_builder = PostEmbedding()

    def from_files(self, post_tags_path: str, tag_vocab: str):
        pass

    def from_db(self):
        post_body_series = pd.read_sql_query("SELECT Body FROM Post WHERE (Tags LIKE '%python%') AND (Body LIKE '%import%')  LIMIT 100000", self.db)

        modules_series = post_body_series['Body'].apply(lambda html: [x.module for x in self.emb_builder.get_imports_via_regex(BeautifulSoup(html, 'lxml'))])
        self.module_vocab = list(set(modules_series.sum()))

        combinations = modules_series.apply(lambda row: list(itertools.combinations(row, 2)))
        combinations = combinations[combinations.astype(str) != '[]']

        # Now concatenate all the lists together
        module_pairs = []
        for i in combinations:
            module_pairs += i
        self.training_pairs = module_pairs

    def sample_n(self, df, train_size: int):
        return random.sample(df, train_size)

    def train(self, train_size: int, epochs: int):
        # Loss
        loss_function = nn.NLLLoss()
        losses = []
        # Model
        self.model = ModuleEmbedding(vocab_size=len(self.module_vocab), embedding_dim=self.emb_size).to(self.device)
        # Optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        # Enumerate the vocabulary, reflects the index of where the 1 is in the one-hot
        self.tag_to_ix = {tag: i for i, tag in enumerate(self.module_vocab)}
        # Reduce size of training set
        samples = self.sample_n(self.training_pairs, train_size)

        for epoch in range(epochs):
            total_loss = 0
            for tag_a, tag_b in tqdm(samples):
                tag_a_id = torch.tensor(self.tag_to_ix[tag_a], dtype=torch.long).to(self.device)
                self.model.zero_grad()
                log_probs = self.model(tag_a_id)
                loss = loss_function(log_probs.flatten(), torch.tensor(self.tag_to_ix[tag_b], dtype=torch.long).to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss)

    def get_tag_embedding(self, tag: str):
        return self.model.embedding.weight[self.tag_to_ix[tag]]

    def to_tensorboard(self, run_name: str):
        """
        Write embedding to Tensorboard projector
        tensorboard --logdir="runs/run@20221102-173048"
        """
        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_embedding(self.model.embedding.weight,
                             metadata=self.module_vocab,
                             tag=f'Next-Tag embedding')
        writer.close()

    def load_model(self, model_path: str, vocab_size: int, embedding_dim: int):
        self.model = ModuleEmbedding(vocab_size, embedding_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)


class ModuleEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(ModuleEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


if __name__ == '__main__':
    met = ModuleEmbeddingTrainer(emb_size=30, database_path='../stackoverflow.db')
    met.from_db()
    print(len(met.training_pairs))
    print(len(met.module_vocab))


    #tet = NextTagEmbeddingTrainer(context_length=3, emb_size=50)

    #tet.from_files("../data/raw/all_tags.csv", "../data/raw/tag_vocab.csv")
    # assert len(tet.post_tags) == 84187510, "Incorrect number of post tags!"
    # assert len(tet.tag_vocab) == 63653, "Incorrect vocab size!"

    met.train(1000, 1)
    # tet.to_tensorboard(f"run@{time.strftime('%Y%m%d-%H%M%S')}")

    # tet.save_model("25mil.pt")
    # tet.load_model("10mil_500d_embd.pt", 63653, 500)
    # tet.to_tensorboard(f"run@{time.strftime('%Y%m%d-%H%M%S')}")