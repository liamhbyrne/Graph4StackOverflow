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
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('tag', 'describes', 'question'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'answer'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'comment'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'question'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'answer'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('question', 'rev_describes', 'tag'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_describes', 'tag'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('comment', 'rev_describes', 'tag'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('question', 'rev_imported_in', 'module'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_imported_in', 'module'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(-1, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        # print("IN", post_emb.shape)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        outs = []
        for x, batch in zip(x_dict.values(), batch_dict.values()):
            if len(x):
                outs.append(global_mean_pool(x, batch=batch, size=len(post_emb)))
            else:
                outs.append(torch.zeros(1, x.size(-1)))

        # print([x.shape for x in outs])
        out = torch.cat(outs, dim=1)

        out = torch.cat([out, post_emb], dim=1)

        # print("B4 LINEAR", out.shape)
        out = self.lin(out)
        out = out.relu()
        out = self.softmax(out)
        return out


'''

'''


def train(model, train_loader):
    running_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data.to(device)

        optimizer.zero_grad()  # Clear gradients.
        
        if INCLUDE_ANSWER:
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            post_emb = data.question_emb.to(device)

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        running_loss += loss.item()
        if i % 5 == 0:
            log.info(f"[{i + 1}] Loss: {running_loss / 5}")
            running_loss = 0.0


def test(loader):
    table = wandb.Table(columns=["ground_truth", "prediction"]) if use_wandb else None
    model.eval()

    predictions = []
    true_labels = []

    loss_ = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        
        if INCLUDE_ANSWER:
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            post_emb = data.question_emb.to(device)

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        loss_ += loss.item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        predictions += list([x.item() for x in pred])
        true_labels += list([x.item() for x in data.label])
        # log.info([(x, y) for x,y in zip([x.item() for x in pred], [x.item() for x in data.label])])
        if use_wandb:
            #graph_html = wandb.Html(plotly.io.to_html(create_graph_vis(data)))
            
            for pred, label in zip(pred, torch.squeeze(data.label, -1)):
                table.add_data(label, pred)

            

    #print([(x, y) for x, y in zip(predictions, true_labels)])
    test_results = {
        "accuracy": accuracy_score(true_labels, predictions),
        "f1-score": f1_score(true_labels, predictions),
        "loss": loss_ / len(loader),
        "table": table,
        "preds": predictions, 
        "trues": true_labels 
    }
    return test_results


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

        # graph_vis = plotly.io.to_html(fig, full_html=False)

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
    # wandb.use_artifact("static-graphs:latest")


def save_model(model, model_name: str):
    torch.save(model.state_dict(), os.path.join("..", "models", model_name))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")

    in_memory_dataset = True
    # Datasets
    if in_memory_dataset:
        train_dataset = UserGraphDatasetInMemory(root="../data", file_name_out='train-4175-qs.pt')
        test_dataset = UserGraphDatasetInMemory(root="../data", file_name_out='test-1790-qs.pt')
    else:
        dataset = UserGraphDataset(root="../data", skip_processing=True)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - (train_size + val_size)


        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    log.info(f"Train Dataset Size: {len(train_dataset)}")
    log.info(f"Test Dataset Size: {len(test_dataset)}")
    
    # Weights&Biases dashboard
    data_details = {
        "num_node_features": train_dataset.num_node_features,
        "num_classes": 2
    }
    log.info(f"Data Details:\n{data_details}")
    
    log.info(train_dataset[0])
    
    setup_wandb = False
    wandb_project_name = "heterogeneous-GAT-model"
    if setup_wandb:
        init_wandb(wandb_project_name, dataset)
    use_wandb = True
    if use_wandb:
        wandb_run_name = f"run@{time.strftime('%Y%m%d-%H%M%S')}"
        start_wandb_for_training(wandb_project_name, wandb_run_name)

    calculate_class_weights = True
    # Class weights
    sampler = None
    if calculate_class_weights:
        log.info(f"Calculating class weights")
        train_labels = [x.label for x in train_dataset]
        counts = [train_labels.count(x) for x in [0, 1]]
        print(counts)
        class_weights = [1 - (x / sum(counts)) for x in counts]
        print(class_weights)
        sampler = torch.utils.data.WeightedRandomSampler([class_weights[x] for x in train_labels], len(train_labels))

    TRAIN_BATCH_SIZE = 512
    log.info(f"Train DataLoader batch size is set to {TRAIN_BATCH_SIZE}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=TRAIN_BATCH_SIZE, num_workers=14)
    
    test_loader = DataLoader(test_dataset, batch_size=512, num_workers=14)

    # Model
    model = HeteroGNN(hidden_channels=64, out_channels=2, num_layers=3)
    model.to(device)
    
    # Experiment config
    INCLUDE_ANSWER = False
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 40):
        log.info(f"Epoch: {epoch:03d} > > >")
        train(model, train_loader)
        train_info = test(train_loader)
        test_info = test(test_loader)

        print(f'Epoch: {epoch:03d}, Train F1: {train_info["f1-score"]:.4f}, Test F1: {test_info["f1-score"]:.4f}')
        checkpoint_file_name = f"../models/model-{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_file_name)
        if use_wandb:
            wandb.log({
                "train/loss": train_info["loss"],
                "train/accuracy": train_info["accuracy"],
                "train/f1": train_info["f1-score"],
                "train/table": train_info["table"],
                "test/loss": test_info["loss"],
                "test/accuracy": test_info["accuracy"],
                "test/f1": test_info["f1-score"],
                "test/table": test_info["table"]
            })
            # Log model checkpoint as an artifact to W&B
            # artifact = wandb.Artifact(name="heterogenous-GAT-static-graphs", type="model")
            # checkpoint_file_name = f  "../models/model-{epoch}.pt"
            # torch.save(model.state_dict(), checkpoint_file_name)
            # artifact.add_file(checkpoint_file_name)
            # wandb.log_artifact(artifact)

    print(f'Test F1: {train_info["f1-score"]:.4f}')

    save_model(model, "model.pt")
    if use_wandb:
        wandb.log({"test/cm": wandb.plot.confusion_matrix(probs=None, y_true=test_info["trues"], preds=test_info["preds"], class_names=["neutral", "upvoted"])})
        wandb.finish()

