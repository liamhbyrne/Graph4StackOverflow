import logging

import torch
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATConv, Linear, global_mean_pool

from dataset import UserGraphDataset

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger('heterogeneous-GAT-model')

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('tag', 'describes', 'question') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'answer') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'comment') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'question') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'answer') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('question', 'rev_describes', 'tag') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_describes', 'tag') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('comment', 'rev_describes', 'tag') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('question', 'rev_imported_in', 'module') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_imported_in', 'module') : GATConv((-1,-1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(-1, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb=2):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        outs = []
        for x, batch in zip(x_dict.values(), batch_dict.values()):
            if len(x):
                outs.append(global_mean_pool(x, batch))
            else:
                outs.append(torch.zeros(1, x.size(-1)))
        out = torch.cat(outs, dim=-1)
        out = torch.cat([torch.squeeze(out), post_emb], dim=-1)
        out = self.lin(out)
        out = out.relu()
        out = self.softmax(out)
        return out



def train(model, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        print(data)
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, torch.concat([data.question_emb, data.answer_emb]))  # Perform a single forward pass.
        loss = criterion(torch.unsqueeze(out,0), data.label)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    predictions = []
    true_labels = []

    model.eval()
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, torch.concat([data.question_emb, data.answer_emb]))
        pred = torch.unsqueeze(out,0).argmax(dim=1)  # Use the class with highest probability.
        predictions.append(pred.item())
        true_labels.append(data.label.item())

    return accuracy_score(true_labels, predictions), f1_score(true_labels, predictions)  # Derive ratio of correct predictions.


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")
    # Datasets
    dataset = UserGraphDataset(root="../data", skip_processing=True)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.1, 0.3])

    # Class weights
    train_labels = [x.label for x in train_dataset]
    counts = [train_labels.count(x) for x in [0,1]]
    class_weights = [1 - (x / sum(counts)) for x in counts]
    sampler = torch.utils.data.WeightedRandomSampler([class_weights[x] for x in train_labels], len(train_labels))

    # Dataloaders
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Model
    model = HeteroGNN(hidden_channels=64, out_channels=2, num_layers=3)
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 10):
        train(model, train_loader)
        train_acc, train_f1 = test(train_loader)
        val_acc, val_f1 = test(val_loader)
        print(f'Epoch: {epoch:03d}, Train F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}')
