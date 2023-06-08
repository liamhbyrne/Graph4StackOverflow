import itertools
import logging
import pickle
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

    def __init__(self, context_length: int, emb_size: int, excluded_tags=None, database_path: str = None):
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
        self.excluded_tags = excluded_tags


    def build_cbow(self, tags: List[str], context_len: int) -> List[Tuple]:

        filtered_tags = [t for t in tags if t not in self.excluded_tags]

        if len(filtered_tags) <= 1:
            return []

        pairs = []

        for target in filtered_tags:
            context = [t for t in filtered_tags if t != target]
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
        post_tags = pd.read_sql_query(f"SELECT Tags FROM Post WHERE PostTypeId=1 AND Tags LIKE '%python%' LIMIT 100000", self.db)
        tag_list_df = post_tags['Tags'].map(self.parse_tag_list)

        self.tag_vocab = list(set(tag_list_df.sum() + ["PAD"]))

        context_and_target = tag_list_df.apply(lambda row: self.build_cbow(row, self.context_length))
        context_and_target = context_and_target[context_and_target.astype(str) != '[]']
        # Now concatenate all the lists together
        tag_pairs = []
        for i in context_and_target:
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

    @staticmethod
    def load_model(model_path: str, vocab_size: int, embedding_dim: int, context_length: int):
        model = NextTagEmbedding(vocab_size, embedding_dim, context_length)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # unpickle the tag_to_ix
        with open(model_path.replace('tag-emb', f'tag_to_ix_tag-emb'), 'rb') as f:
            model.tag_to_ix = pickle.load(f)

        return model

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)
        # pickle the tag_to_ix
        with open('tag_to_ix_' + model_path, 'wb') as f:
            pickle.dump(self.tag_to_ix, f)


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

    def get_tag_embedding(self, tag: str):
        assert tag in self.tag_to_ix, "Tag not in vocabulary!"
        assert self.tag_to_ix is not None, "Tag to index mapping not set!"
        return self.embedding.weight[self.tag_to_ix[tag]]


if __name__ == '__main__':
    tet = NextTagEmbeddingTrainer(context_length=2, emb_size=30, excluded_tags=['python'], database_path="../stackoverflow.db")
    #tet.from_db()
    #print(len(tet.post_tags))
    #print(len(tet.tag_vocab))


    #tet = NextTagEmbeddingTrainer(context_length=3, emb_size=50)

    tet.from_files("../all_tags.csv", "../tag_vocab.csv")
    # assert len(tet.post_tags) == 84187510, "Incorrect number of post tags!"
    # assert len(tet.tag_vocab) == 63653, "Incorrect vocab size!"
    
    print(len(tet.post_tags))

    tet.train(1000, 1)
    # tet.to_tensorboard(f"run@{time.strftime('%Y%m%d-%H%M%S')}")

    # tet.save_model("25mil.pt")
    # tet.load_model("10mil_500d_embd.pt", 63653, 500)
    # tet.to_tensorboard(f"run@{time.strftime('%Y%m%d-%H%M%S')}")
