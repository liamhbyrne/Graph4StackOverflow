import logging
import os
import pickle
import random
import sqlite3
from typing import List, Tuple

import pandas as pd
import torch
from torch import nn, optim
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ACL2024.modules.util.get_root_dir import get_project_root

logging.basicConfig(level=logging.INFO)


class NextTagEmbeddingTrainer:
    def __init__(
        self,
        context_length: int,
        emb_size: int,
        excluded_tags=None,
        database_path: str = None,
    ):
        """
        Initialize the Next Tag Embedding Trainer.

        Args:
        - context_length (int): Length of the context window.
        - emb_size (int): Size of the embeddings.
        - excluded_tags (List[str]): Tags to be excluded from training.
        - database_path (str): Path to the database (if applicable).
        """
        # Set up logging and device
        logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Proceeding with {self.device} . .")

        # Connect to the database if path is provided
        if database_path is not None:
            self.db = sqlite3.connect(database_path)
            logger.info(f"Connected to {database_path}")

        # Initialize variables
        self.tag_vocab: List[str]
        self.post_tags: List[Tuple]
        self.context_length = context_length
        self.emb_size = emb_size
        self.excluded_tags = excluded_tags

    def build_cbow(self, tags: List[str], context_len: int) -> List[Tuple]:
        """
        Build bags of words, with negative sampling
        """
        # Filtering tags and creating context-target pairs
        filtered_tags = [t for t in tags if t not in self.excluded_tags]

        if len(filtered_tags) <= 1:
            return []

        pairs = []

        for target in filtered_tags:
            context = [t for t in filtered_tags if t != target]
            # Pad or cut based on the context length
            while len(context) < context_len:
                context.append("PAD")
            while len(context) > context_len:
                context = context[:-1]
            pairs.append((context, target))

        return pairs

    def negative_sampling(self, target, num_samples):
        """
        Performs negative sampling for a target word.

        Args:
        - target (str): The target word for which negative samples are generated.
        - num_samples (int): Number of negative samples to generate.

        Returns:
        - List[str]: List of negative samples.
        """
        # Choose random words as negative samples that are not the target
        negative_samples = []
        while len(negative_samples) < num_samples:
            neg_sample = random.choice(self.tag_vocab)
            if neg_sample != target:
                negative_samples.append(neg_sample)
        return negative_samples

    def from_files(self, post_tags_path: str, tag_vocab: str):
        """
        Extracts data from files to build training data.

        Args:
        - post_tags_path (str): Path to the post tags file.
        - tag_vocab (str): Path to the tag vocabulary file.
        """
        # Reading tag vocabulary
        tag_df = pd.read_csv(tag_vocab, keep_default_na=False)
        self.tag_vocab = list(set(tag_df["TagName"])) + ["PAD"]

        # Processing post tags
        post_tags = pd.read_csv(post_tags_path)
        tag_list_df = post_tags["Tags"].apply(
            lambda row: self.build_cbow(self.parse_tag_list(row), self.context_length)
        )
        context_and_target = tag_list_df[tag_list_df.astype(str) != "[]"]

        # Concatenating all lists together
        tag_pairs = []
        for i in context_and_target:
            tag_pairs += i
        self.post_tags = tag_pairs

    def from_db(self):
        """
        Extracts data from a database to build training data.
        """
        # Retrieving tags from the database
        post_tags = pd.read_sql_query(
            "SELECT Tags FROM Post WHERE PostTypeId=1 AND Tags LIKE '%python%' LIMIT 100000",
            self.db,
        )
        tag_list_df = post_tags["Tags"].map(self.parse_tag_list)

        # Generating tag vocabulary and context-target pairs
        self.tag_vocab = list(set(tag_list_df.sum() + ["PAD"]))

        context_and_target = tag_list_df.apply(
            lambda row: self.build_cbow(row, self.context_length)
        )
        context_and_target = context_and_target[context_and_target.astype(str) != "[]"]

        # Concatenating all lists together
        tag_pairs = []
        for i in context_and_target:
            tag_pairs += i
        self.post_tags = tag_pairs

    def parse_tag_list(self, tag_list: str) -> List[str]:
        """
        Parses a string of tags into a list of strings.

        Args:
        - tag_list (str): String containing tags.

        Returns:
        - List[str]: List of tags.
        """
        return tag_list[1:-1].split("><")

    def sample_n(self, df, train_size: int):
        """
        Samples a specified size from the dataset.

        Args:
        - df: Data to sample from.
        - train_size (int): Size of the sample.

        Returns:
        - List: Sampled data.
        """
        return random.sample(df, train_size)

    def train(self, train_size: int, epochs: int, num_negative_samples: int = 5):
        """
        Trains the Next Tag Embedding model using CBOW with negative sampling.

        Args:
        - train_size (int): Size of the training dataset.
        - epochs (int): Number of epochs for training.
        - num_negative_samples (int): Number of negative samples to use for negative sampling.
        """
        # Loss function for binary classification
        loss_function = nn.BCEWithLogitsLoss()
        losses = []

        # Model initialization
        self.model = NextTagEmbedding(
            vocab_size=len(self.tag_vocab),
            embedding_dim=self.emb_size,
            context_size=self.context_length,
        ).to(self.device)

        # Optimizer setup
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        # Enumerate the vocabulary, reflecting the index of where the 1 is in the one-hot
        self.tag_to_ix = {tag: i for i, tag in enumerate(self.tag_vocab)}

        # Reduce size of training set
        samples = self.sample_n(self.post_tags, train_size)

        for epoch in range(epochs):
            total_loss = 0
            for context, target in tqdm(samples):
                context_tensor = torch.tensor(
                    [self.tag_to_ix[t] for t in context], dtype=torch.long
                ).to(self.device)

                self.model.zero_grad()

                target_idx = torch.tensor(self.tag_to_ix[target], dtype=torch.long).to(
                    self.device
                )

                # Perform negative sampling to obtain negative examples
                negative_samples = self.negative_sampling(target, num_negative_samples)
                neg_samples_indices = [
                    self.tag_to_ix[sample] for sample in negative_samples
                ]

                # Combine positive and negative samples
                all_samples = [target_idx] + neg_samples_indices

                # Forward pass, target labels, loss calculation, and backpropagation
                log_probs = self.model(context_tensor)
                output = log_probs.squeeze(0)
                targets = torch.cat(
                    [torch.tensor([1.0]), torch.tensor([0.0] * num_negative_samples)]
                )

                loss = loss_function(output[all_samples], targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            losses.append(total_loss)

    def to_tensorboard(self, run_name: str):
        """
        Writes embeddings to Tensorboard projector.

        Args:
        - run_name (str): Name for the Tensorboard run.
        """
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_embedding(
            self.model.embedding.weight,
            metadata=self.tag_vocab,
            tag="Next-Tag embedding",
        )
        writer.close()

    @staticmethod
    def load_model(
        model_path: str, vocab_size: int, embedding_dim: int, context_length: int
    ):
        """
        Loads a pre-trained model from a specified path.

        Args:
        - model_path (str): Path to the saved model.
        - vocab_size (int): Size of the vocabulary used for the model.
        - embedding_dim (int): Dimension of the embeddings.
        - context_length (int): Length of the context window.

        Returns:
        - NextTagEmbedding: The loaded model.
        """
        # Join the model path to the project root
        model_path = os.path.join(get_project_root(), model_path)

        model = NextTagEmbedding(vocab_size, embedding_dim, context_length)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

        # Unpickle the tag_to_ix
        with open(model_path.replace("tag-emb", "tag_to_ix_tag-emb"), "rb") as f:
            model.tag_to_ix = pickle.load(f)

        return model

    def save_model(self, model_path: str):
        """
        Saves the trained model to a specified path.

        Args:
        - model_path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), model_path)
        # Pickle the tag_to_ix
        with open("tag_to_ix_" + model_path, "wb") as f:
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
        assert tag in self.tag_to_ix, f"{tag} - Tag not in vocabulary!"
        assert self.tag_to_ix is not None, "Tag to index mapping not set!"
        return self.embedding.weight[self.tag_to_ix[tag]]


if __name__ == "__main__":
    # Create an instance of NextTagEmbeddingTrainer
    tet = NextTagEmbeddingTrainer(
        context_length=2,
        emb_size=30,
        excluded_tags=["python"],
        database_path="../stackoverflow.db",
    )

    # Initializing and training the model
    tet.from_files(
        os.path.join(get_project_root(), "modules", "dataset", "all_tags.csv"),
        os.path.join(get_project_root(), "modules", "dataset", "tag_vocab.csv"),
    )

    print(len(tet.post_tags))
    tet.train(1000, 1)

    # Saving and loading the model
    tet.save_model("TODO.pt")
    tet.load_model("TODO.pt", 63653, 500, context_length=3)
