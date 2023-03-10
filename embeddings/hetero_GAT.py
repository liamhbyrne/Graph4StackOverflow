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
from helper_functions import calculate_class_weights, split_test_train_pytorch
import wandb
from torch_geometric.utils import to_networkx
from sklearn.model_selection import KFold

from custom_logger import setup_custom_logger
from dataset import UserGraphDataset
from dataset_in_memory import UserGraphDatasetInMemory
from Visualize import GraphVisualization
import helper_functions
from hetero_GAT_constants import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, IN_MEMORY_DATASET, INCLUDE_ANSWER, USE_WANDB, WANDB_PROJECT_NAME, NUM_WORKERS, EPOCHS, NUM_LAYERS, HIDDEN_CHANNELS, FINAL_MODEL_OUT_PATH, SAVE_CHECKPOINTS, WANDB_RUN_NAME

log = setup_custom_logger("heterogenous_GAT_model", logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


class HeteroGNN(torch.nn.Module):
    """
    Heterogenous Graph Attentional Network (GAT)
    """
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        # Create Graph Attentional layers
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
    table = wandb.Table(columns=["ground_truth", "prediction"]) if USE_WANDB else None
    model.eval()

    predictions = []
    true_labels = []

    cumulative_loss = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        
        if INCLUDE_ANSWER:
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            post_emb = data.question_emb.to(device)

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        cumulative_loss += loss.item()
        
        # Use the class with highest probability.
        pred = out.argmax(dim=1)  
        
        # Cache the predictions for calculating metrics
        predictions += list([x.item() for x in pred])
        true_labels += list([x.item() for x in data.label])
        
        # Log table of predictions to WandB
        if USE_WANDB:
            #graph_html = wandb.Html(plotly.io.to_html(create_graph_vis(data)))
            
            for pred, label in zip(pred, torch.squeeze(data.label, -1)):
                table.add_data(label, pred)
    
    # Collate results into a single dictionary
    test_results = {
        "accuracy": accuracy_score(true_labels, predictions),
        "f1-score": f1_score(true_labels, predictions),
        "loss": cumulative_loss / len(loader),
        "table": table,
        "preds": predictions, 
        "trues": true_labels 
    }
    return test_results



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")

    # Datasets
    if IN_MEMORY_DATASET:
        train_dataset = UserGraphDatasetInMemory(root="../data", file_name_out='train-4175-qs.pt')
        test_dataset = UserGraphDatasetInMemory(root="../data", file_name_out='test-1790-qs.pt')
        
        ## TEST
        split1, split2 = helper_functions.split_test_train_pytorch(train_dataset, 0.7)

    else:
        dataset = UserGraphDataset(root="../data", skip_processing=True)
        train_dataset, test_dataset = split_test_train_pytorch(dataset)

    log.info(f"Sample graph:\n{train_dataset[0]}")
    log.info(f"Train Dataset Size: {len(train_dataset)}")
    log.info(f"Test Dataset Size: {len(test_dataset)}")
    
    # Weights&Biases dashboard
    data_details = {
        "num_node_features": train_dataset.num_node_features,
        "num_classes": 2
    }
    log.info(f"Data Details:\n{data_details}")
    
    if USE_WANDB:
        if WANDB_RUN_NAME is None:
            run_name = f"run@{time.strftime('%Y%m%d-%H%M%S')}"
        helper_functions.start_wandb_for_training(WANDB_PROJECT_NAME, run_name)


    # Class weights
    sampler = calculate_class_weights(train_dataset)

    # Dataloaders
    log.info(f"Train DataLoader batch size is set to {TRAIN_BATCH_SIZE}")
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS)

    # Model
    model = HeteroGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=2, num_layers=NUM_LAYERS)
    model.to(device)  # To GPU if available
    
    # Optimizers & Loss function
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(1, EPOCHS):
        log.info(f"Epoch: {epoch:03d} > > >")
        
        # train model . . 
        train(model, train_loader)

        # evaluate on training set . . 
        train_info = test(train_loader)

        # evaluate on test set . . 
        test_info = test(test_loader)

        # log for current epoch
        log.info(f'Epoch: {epoch:03d}, Train F1: {train_info["f1-score"]:.4f}, Test F1: {test_info["f1-score"]:.4f}')

        if SAVE_CHECKPOINTS:
            checkpoint_file_name = f"../models/model-{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_file_name)
        
        # log evaluation results to wandb
        if USE_WANDB:
            helper_functions.log_results_to_wandb(train_info, "train")
            helper_functions.log_results_to_wandb(test_info, "test")

    log.info(f'Test F1: {train_info["f1-score"]:.4f}')

    helper_functions.save_model(model, FINAL_MODEL_OUT_PATH)
    # Plot confusion matrix
    if USE_WANDB:
        helper_functions.add_cm_to_wandb(test_info)
        wandb.finish()
