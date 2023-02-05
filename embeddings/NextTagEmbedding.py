import itertools
import logging
import random
import sqlite3
from typing import *

import pandas as pd
import torch
from torch import nn, optim
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class NextTagEmbeddingTrainer:

    def __init__(self, context_length: int, emb_size: int, database_path: str = None):
        logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Proceeding with {self.device} . .")
        if database_path is not None:
            self.db = sqlite3.connect(database_path)
            logger.info(f"Connected to {database_path}")
        self.tag_vocab: List[str]
        self.post_tags: List[Tuple]
        self.context_length = context_length
        self.emb_size = emb_size

    def build_cbow(self, tags: List[str], context_len: int) -> List[Tuple]:
        pairs = []
        if len(tags) <= 1:
            return []

        for target in tags:
            context = [t for t in tags if t != target]
            # Pad or cut depending on the context length
            while len(context) < context_len:
                context.append('PAD')
            while len(context) > context_len:
                context = context[:-1]
            pairs.append((context, target))

        return pairs

    def from_files(self, post_tags_path: str, tag_vocab: str):
        tag_df = pd.read_csv(tag_vocab, keep_default_na=False)
        self.tag_vocab = list(set(tag_df["TagName"])) + ["PAD"]

        post_tags = pd.read_csv(post_tags_path)
        tag_list_df = post_tags['Tags'].apply(lambda row: self.build_cbow(self.parse_tag_list(row), self.context_length))
        context_and_target = tag_list_df[tag_list_df.astype(str) != '[]']
        # Now concatenate all the lists together
        tag_pairs = []
        for i in context_and_target:
            tag_pairs += i
        self.post_tags = tag_pairs

    def from_db(self):
        tag_df = pd.read_sql_query("SELECT * FROM Tag", self.db)
        tag_df.set_index('TagId', inplace=True)
        self.tag_vocab = list(set(tag_df["TagName"]))

        post_tags = pd.read_sql_query(f"SELECT Tags FROM Post WHERE PostTypeId=1", self.db)
        tag_list_df = post_tags['Tags'].map(self.parse_tag_list)
        combinations = tag_list_df.apply(lambda row: list(itertools.combinations(row, 2)))
        combinations = combinations[combinations.astype(str) != '[]']
        # Now concatenate all the lists together
        tag_pairs = []
        for i in combinations:
            tag_pairs += i
        self.post_tags = tag_pairs

    def parse_tag_list(self, tag_list: str) -> List[str]:
        return tag_list[1:-1].split("><")

    def sample_n(self, df, train_size: int):
        return random.sample(df, train_size)

    def train(self, train_size: int, epochs: int):
        # Loss
        loss_function = nn.NLLLoss()
        losses = []
        # Model
        self.model = NextTagEmbedding(vocab_size=len(self.tag_vocab), embedding_dim=self.emb_size, context_size=self.context_length).to(self.device)
        # Optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        # Enumerate the vocabulary, reflects the index of where the 1 is in the one-hot
        self.tag_to_ix = {tag: i for i, tag in enumerate(self.tag_vocab)}
        # Reduce size of training set
        samples = self.sample_n(self.post_tags, train_size)

        for epoch in range(epochs):
            total_loss = 0
            for context, target in tqdm(samples):
                context_tensor = torch.tensor([self.tag_to_ix[t] for t in context], dtype=torch.long).to(self.device)
                self.model.zero_grad()
                print(context_tensor)
                log_probs = self.model(context_tensor)
                loss = loss_function(log_probs.flatten(), torch.tensor(self.tag_to_ix[target], dtype=torch.long).to(self.device))
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
                             metadata=self.tag_vocab,
                             tag=f'Next-Tag embedding')
        writer.close()

    def load_model(self, model_path: str, vocab_size: int, embedding_dim: int):
        self.model = NextTagEmbedding(vocab_size, embedding_dim, self.context_length)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)


class NextTagEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NextTagEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


if __name__ == '__main__':
    # tet = TagEmbeddingTrainer("../stackoverflow.db")
    # tet.from_db()
    tet = NextTagEmbeddingTrainer(context_length=3, emb_size=30)

    tet.from_files("../data/raw/all_tags.csv", "../data/raw/tag_vocab.csv")
    # assert len(tet.post_tags) == 84187510, "Incorrect number of post tags!"
    # assert len(tet.tag_vocab) == 63653, "Incorrect vocab size!"

    tet.train(100, 1)
    # tet.to_tensorboard(f"run@{time.strftime('%Y%m%d-%H%M%S')}")

    # tet.save_model("25mil.pt")
    # tet.load_model("10mil_500d_embd.pt", 63653, 500)
    # tet.to_tensorboard(f"run@{time.strftime('%Y%m%d-%H%M%S')}")
