import networkx as nx
import torch
from torch_geometric.utils import to_networkx

import wandb
import os

from Visualize import GraphVisualization
from custom_logger import setup_custom_logger
import logging

log = setup_custom_logger("heterogenous_GAT_model", logging.INFO)


def create_graph_vis(graph):
    g = to_networkx(graph.to_homogeneous())
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g,
        pos,
        node_text_position="top left",
        node_size=20,
    )
    fig = vis.create_figure()
    return fig


"""
Weights & Biases dashboard
"""


def init_wandb(project_name: str, dataset, data_details):
    wandb.init(project=project_name, name="setup")

    # Log all the details about the data to W&B.
    wandb.log(data_details)

    # Log exploratory visualizations for each data point to W&B
    table = wandb.Table(
        columns=["Graph", "Number of Nodes", "Number of Edges", "Label"]
    )
    for graph in dataset:
        fig = create_graph_vis(graph)
        n_nodes = graph.num_nodes
        n_edges = graph.num_edges
        label = graph.label.item()

        # graph_vis = plotly.io.to_html(fig, full_html=False)

        table.add_data(wandb.Plotly(fig), n_nodes, n_edges, label)
    wandb.log({"data": table})

    # Log the dataset to W&B as an artifact.
    dataset_artifact = wandb.Artifact(
        name="static-graphs", type="dataset", metadata=data_details
    )
    dataset_artifact.add_dir("../data/")
    wandb.log_artifact(dataset_artifact)

    # End the W&B run
    wandb.finish()


def start_wandb_for_training(wandb_project_name: str, wandb_run_name: str):
    wandb.init(project=wandb_project_name, name=wandb_run_name)
    return wandb.config


def add_cm_to_wandb(test_info):
    wandb.log(
        {
            "test/cm": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_info["trues"],
                preds=test_info["preds"],
                class_names=["neutral", "upvoted"],
            )
        }
    )


def log_results_to_wandb(results_map, results_name: str):
    wandb.log(
        {
            f"{results_name}/loss": results_map["loss"],
            f"{results_name}/accuracy": results_map["accuracy"],
            f"{results_name}/f1-macro": results_map["f1-score-macro"],
            f"{results_name}/f1-weighted": results_map["f1-score-weighted"],
            f"{results_name}/table": results_map["table"],
        }
    )


"""
PyTorch helpers
"""


def save_model(model, model_name: str):
    torch.save(model.state_dict(), os.path.join("..", "models", model_name))


def split_test_train_pytorch(dataset, train_split):
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - (train_size)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


def calculate_class_weights(dataset):
    # Class weights
    log.info("Calculating class weights")
    train_labels = [x.label for x in dataset]
    counts = [train_labels.count(x) for x in [0, 1]]
    class_weights = [1 - (x / sum(counts)) for x in counts]
    log.info(class_weights)

    return torch.tensor(class_weights)
