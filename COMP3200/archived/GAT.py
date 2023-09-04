import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, Linear, MeanAggregation, to_hetero

from dataset import UserGraphDataset

metadata = (
    ["question", "answer", "comment", "tag", "module"],
    [
        ("tag", "describes", "question"),
        ("tag", "describes", "answer"),
        ("tag", "describes", "comment"),
        ("module", "imported_in", "question"),
        ("module", "imported_in", "answer"),
        ("question", "rev_describes", "tag"),
        ("answer", "rev_describes", "tag"),
        ("comment", "rev_describes", "tag"),
        ("question", "rev_imported_in", "module"),
        ("answer", "rev_imported_in", "module"),
    ],
)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Model, self).__init__()
        self.lin = Linear(hidden_channels, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.gat = to_hetero(GAT(hidden_channels), metadata)
        self.pool = MeanAggregation()

    def forward(self, data, post_emb):
        convolved = self.gat(data.x_dict, data.edge_index_dict, data.batch_dict)
        pooled = self.pool(convolved, data.batch_dict["question"])

        x = torch.cat([convolved, post_emb], dim=1)

        # 3. Concatenate with post embedding
        # x = torch.cat((x, post_emb))
        # 4. Apply a final classifier.
        x = self.lin(x)
        x = self.softmax(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv3 = GATConv((-1, -1), hidden_channels, add_self_loops=False)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        # 2. Readout layer
        x = self.pool(x, batch)  # [batch_size, hidden_channels]
        return x


def train(model, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        print(data)

        out = model(
            data, torch.cat([data.question_emb, data.answer_emb], dim=1)
        )  # Perform a single forward pass.
        print(out, data.label)
        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == "__main__":
    dataset = UserGraphDataset(root="../data", skip_processing=True)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.6, 0.1, 0.3]
    )

    print(dataset.num_node_features)

    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = Model(hidden_channels=64)
    sample = train_dataset[0]
    # model = to_hetero(model, metadata)
    # print(model(sample.x_dict, sample.edge_index_dict, sample.batch_dict))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 10):
        train(model, train_loader)
        # train_acc = test(train_loader)
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
