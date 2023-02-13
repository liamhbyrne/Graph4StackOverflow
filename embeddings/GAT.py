import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, Linear, MeanAggregation, to_hetero

from dataset import UserGraphDataset


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv3 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin = Linear(hidden_channels, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.pool = MeanAggregation()

    def forward(self, x, edge_index, batch, post_emb):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = self.pool(x, batch)  # [batch_size, hidden_channels]

        x = torch.concat([x, post_emb])
        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        #x = self.softmax(x)
        return x


def train(model, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        print(data)

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, torch.concat([data.question_emb, data.answer_emb]))  # Perform a single forward pass.
        print(out, data.label)
        loss = criterion(out, data.label)  # Compute the loss.
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


if __name__ == '__main__':
    dataset = UserGraphDataset(root="../data", skip_processing=True)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.1, 0.3])

    print(dataset.num_node_features)

    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = GAT(hidden_channels=64)
    sample = train_dataset[0]
    metadata = (['question', 'answer', 'comment', 'tag', 'module'], [('tag', 'describes', 'question'), ('tag', 'describes', 'answer'), ('tag', 'describes', 'comment'), ('module', 'imported_in', 'question'), ('module', 'imported_in', 'answer'), ('question', 'rev_describes', 'tag'), ('answer', 'rev_describes', 'tag'), ('comment', 'rev_describes', 'tag'), ('question', 'rev_imported_in', 'module'), ('answer', 'rev_imported_in', 'module')])
    model = to_hetero(model, metadata, aggr='sum')
    #print(model(sample.x_dict, sample.edge_index_dict, sample.batch_dict))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 10):
        train(model, train_loader)
        #train_acc = test(train_loader)
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
