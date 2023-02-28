import json
import logging
import os
import string
import time

import networkx as nx
import plotly
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATConv, Linear, global_mean_pool
import wandb
from torch_geometric.utils import to_networkx

from custom_logger import setup_custom_logger
from dataset import UserGraphDataset
from dataset_in_memory import UserGraphDatasetInMemory
from Visualize import GraphVisualization

log = setup_custom_logger("heterogenous_GAT_model", logging.INFO)


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


    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        #print("IN", post_emb.shape)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        outs = []
        for x, batch in zip(x_dict.values(), batch_dict.values()):
            if len(x):
                outs.append(global_mean_pool(x, batch=batch, size=len(post_emb)))
            else:
                outs.append(torch.zeros(1, x.size(-1)))

        #print([x.shape for x in outs])
        out = torch.cat(outs, dim=1)

        out = torch.cat([out, post_emb], dim=1)

        #print("B4 LINEAR", out.shape)
        out = self.lin(out)
        out = out.relu()
        out = self.softmax(out)
        return out



def train(model, train_loader):
    running_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data = data.to(device)

        optimizer.zero_grad()  # Clear gradients.
        #print("DATA IN", data.question_emb.shape, data.answer_emb.shape)
        post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        running_loss += loss.item()
        if i % 5 == 0:
            log.info(f"[{i+1}] Loss: {running_loss / 2000}")
            running_loss = 0.0



def test(loader):
    table = wandb.Table(columns=["graph", "ground_truth", "prediction"]) if use_wandb else None
    model.eval()

    predictions = []
    true_labels = []

    loss_ = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)

        post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        loss_ += loss.item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        predictions += list([x.item() for x in pred])
        true_labels += list([x.item() for x in data.label])

        if use_wandb:
            graph_html = wandb.Html(plotly.io.to_html(create_graph_vis(data)))
            for pred, label in zip(pred, torch.squeeze(data.label, -1)):
                table.add_data(graph_html, label, pred)

    return accuracy_score(true_labels, predictions), f1_score(true_labels, predictions), loss_ / len(loader), table

def create_graph_vis(graph):
    g = to_networkx(graph.to_homogeneous())
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g, pos, node_text_position='top left', node_size=20,
    )
    fig = vis.create_figure()
    return fig

def init_wandb(project_name: str, dataset):
    wandb.init(project=project_name, name="setup")
    # Log all the details about the data to W&B.
    wandb.log(data_details)

    # Log exploratory visualizations for each data point to W&B
    table = wandb.Table(columns=["Graph", "Number of Nodes", "Number of Edges", "Label"])
    for graph in dataset:
        fig = create_graph_vis(graph)
        n_nodes = graph.num_nodes
        n_edges = graph.num_edges
        label = graph.label.item()

        #graph_vis = plotly.io.to_html(fig, full_html=False)

        table.add_data(wandb.Plotly(fig), n_nodes, n_edges, label)
    wandb.log({"data": table})

    # Log the dataset to W&B as an artifact.
    dataset_artifact = wandb.Artifact(name="static-graphs", type="dataset", metadata=data_details)
    dataset_artifact.add_dir("../data/")
    wandb.log_artifact(dataset_artifact)

    # End the W&B run
    wandb.finish()

def start_wandb_for_training(wandb_project_name: str, wandb_run_name: str):
    wandb.init(project=wandb_project_name, name=wandb_run_name)
    #wandb.use_artifact("static-graphs:latest")

def save_model(model, model_name: str):
    torch.save(model.state_dict(), os.path.join("..", "models", model_name))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")

    in_memory_dataset = False
    # Datasets
    if in_memory_dataset:
        dataset = UserGraphDatasetInMemory(root="../data")
    else:
        dataset = UserGraphDataset(root="../data", skip_processing=True)


    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    log.info(f"Train Dataset Size: {train_size}")
    log.info(f"Validation Dataset Size: {val_size}")
    log.info(f"Test Dataset Size: {test_size}")
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Weights&Biases dashboard
    data_details = {
        "num_node_features": dataset.num_node_features,
        "num_classes": 2
    }
    log.info(f"Data Details:\n{data_details}")

    setup_wandb = False
    wandb_project_name = "heterogeneous-GAT-model"
    if setup_wandb:
        init_wandb(wandb_project_name, dataset)
    use_wandb = False
    if use_wandb:
        wandb_run_name = f"run@{time.strftime('%Y%m%d-%H%M%S')}"
        start_wandb_for_training(wandb_project_name, wandb_run_name)


    calculate_class_weights = False
    #Class weights
    sampler = None
    if calculate_class_weights:
        log.info(f"Calculating class weights")
        train_labels = [x.label for x in train_dataset]
        counts = [train_labels.count(x) for x in [0,1]]
        class_weights = [1 - (x / sum(counts)) for x in counts]
        sampler = torch.utils.data.WeightedRandomSampler([class_weights[x] for x in train_labels], len(train_labels))

    # Dataloaders
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Model
    model = HeteroGNN(hidden_channels=64, out_channels=2, num_layers=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 5):
        log.info(f"Epoch: {epoch:03d} > > >")
        train(model, train_loader)
        train_acc, train_f1, train_loss, train_table = test(train_loader)
        val_acc, val_f1, val_loss, val_table = test(val_loader)
        test_acc, test_f1, test_loss, test_table = test(test_loader)

        print(f'Epoch: {epoch:03d}, Train F1: {train_f1:.4f}, Validation F1: {val_f1:.4f} Test F1: {test_f1:.4f}')
        checkpoint_file_name = f"../models/model-{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_file_name)
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/f1": train_f1,
                "train/table": train_table,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
                "val/table": val_table,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "test/f1": test_f1,
                "test/table": test_table,
            })
            # Log model checkpoint as an artifact to W&B
            # artifact = wandb.Artifact(name="heterogenous-GAT-static-graphs", type="model")
            # checkpoint_file_name = f"../models/model-{epoch}.pt"
            # torch.save(model.state_dict(), checkpoint_file_name)
            # artifact.add_file(checkpoint_file_name)
            # wandb.log_artifact(artifact)

    print(f'Test F1: {test_f1:.4f}')

    save_model(model, "model.pt")
    if use_wandb:
        wandb.finish()
