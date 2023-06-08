import json
import logging
import os
import string
import time

import networkx as nx
import pandas as pd
import plotly
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATv2Conv, GATConv, Linear, global_mean_pool, GCNConv, SAGEConv
from helper_functions import calculate_class_weights, split_test_train_pytorch
import wandb
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ExponentialLR
import pickle

from custom_logger import setup_custom_logger
from dataset import UserGraphDataset
from dataset_in_memory import UserGraphDatasetInMemory
from Visualize import GraphVisualization
import helper_functions
from hetero_GAT_constants import OS_NAME, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, IN_MEMORY_DATASET, INCLUDE_ANSWER, USE_WANDB, WANDB_PROJECT_NAME, NUM_WORKERS, EPOCHS, NUM_LAYERS, HIDDEN_CHANNELS, FINAL_MODEL_OUT_PATH, SAVE_CHECKPOINTS, WANDB_RUN_NAME, CROSS_VALIDATE, FOLD_FILES, USE_CLASS_WEIGHTS_SAMPLER, USE_CLASS_WEIGHTS_LOSS, DROPOUT, GAMMA, START_LR, PICKLE_PATH_KF, ROOT, TRAIN_DATA_PATH, TEST_DATA_PATH, WARM_START_FILE, MODEL, REL_SUBSET

log = setup_custom_logger("heterogenous_GAT_model", logging.INFO)

if OS_NAME == "linux":
    torch.multiprocessing.set_sharing_strategy('file_system')
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


"""
G
A
T
"""

class HeteroGAT(torch.nn.Module):
    """
    Heterogeneous Graph Attentional Network (GAT) model.
    """
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        log.info("MODEL: GAT")

        self.convs = torch.nn.ModuleList()

        # Create Graph Attentional layers
        for _ in range(num_layers):
            conv = HeteroConv({
                ('tag', 'describes', 'question'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('tag', 'describes', 'answer'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('tag', 'describes', 'comment'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('module', 'imported_in', 'question'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('module', 'imported_in', 'answer'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('question', 'rev_describes', 'tag'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('answer', 'rev_describes', 'tag'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('comment', 'rev_describes', 'tag'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('question', 'rev_imported_in', 'module'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('answer', 'rev_imported_in', 'module'): GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        x_dict = {key: x_dict[key] for key in x_dict.keys() if key in ["question", "answer", "comment", "tag"]}

        
        for conv in self.convs:
            break
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=DROPOUT, training=self.training) for key, x in x_dict.items()}

        outs = []
        
        for x, batch in zip(x_dict.values(), batch_dict.values()):
            if len(x):
                outs.append(global_mean_pool(x, batch=batch, size=len(post_emb)).to(device))
            else:
                outs.append(torch.zeros(1, x.size(-1)).to(device))


        out = torch.cat(outs, dim=1).to(device)

        out = torch.cat([out, post_emb], dim=1).to(device)
        
        out = F.dropout(out, p=DROPOUT, training=self.training)


        out = self.lin1(out)
        out = F.leaky_relu(out)
        
        out = self.lin2(out)
        out = F.leaky_relu(out)
        
        out = self.softmax(out)
        return out


"""
T
R
A
I
N
"""        
def train_epoch(train_loader):
    running_loss = 0.0
    model.train()

    for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data.to(device)

        optimizer.zero_grad()  # Clear gradients.

        if INCLUDE_ANSWER:
            # Concatenate question and answer embeddings to form post embeddings
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            # Use only question embeddings as post embedding
            post_emb = data.question_emb.to(device)
        post_emb.requires_grad = True

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, post_emb)  # Perform a single forward pass.

        #y = torch.tensor([1 if x > 0 else 0 for x in data.score]).to(device)
        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        running_loss += loss.item()
        if i % 5 == 0:
            log.info(f"[{i + 1}] Loss: {running_loss / 5}")
            running_loss = 0.0

"""
T
E
S
T
"""
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

        #y = torch.tensor([1 if x > 0 else 0 for x in data.score]).to(device)
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
        "f1-score-weighted": f1_score(true_labels, predictions, average='weighted'),
        "f1-score-macro": f1_score(true_labels, predictions, average='macro'),
        "loss": cumulative_loss / len(loader),
        "table": table,
        "preds": predictions, 
        "trues": true_labels 
    }
    return test_results




"""
SWEEP
"""

def build_dataset(train_batch_size):
    train_dataset = UserGraphDatasetInMemory(root=ROOT, file_name_out=TRAIN_DATA_PATH, question_ids=[])
    test_dataset = UserGraphDatasetInMemory(root=ROOT, file_name_out=TEST_DATA_PATH, question_ids=[])

    class_weights = calculate_class_weights(train_dataset).to(device)
    train_labels = [x.label for x in train_dataset]
    sampler = torch.utils.data.WeightedRandomSampler([class_weights[x] for x in train_labels], len(train_labels))
    
    
    # Dataloaders
    log.info(f"Train DataLoader batch size is set to {TRAIN_BATCH_SIZE}")
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=train_batch_size, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS)
    return train_loader, test_loader

    
def build_network(channels, layers):
    model = HeteroGAT(hidden_channels=channels, out_channels=2, num_layers=layers)
    return model.to(device)
    

    

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, test_loader = build_dataset(config.batch_size)
        
        DROPOUT = config.dropout
        global model
        model = build_network(config.hidden_channels, config.num_layers)
        
        
        # Optimizers & Loss function
        global optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr)
        global scheduler
        scheduler = ExponentialLR(optimizer, gamma=GAMMA, verbose=True)
    
        # Cross Entropy Loss (with optional class weights)
        global criterion
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(config.epochs):
            train_epoch(train_loader)
            f1 = test(test_loader)
            wandb.log({'validation/weighted-f1': f1, "epoch": epoch})    

def test(loader):
    model.eval()

    predictions = []
    true_labels = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        
        if INCLUDE_ANSWER:
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            post_emb = data.question_emb.to(device)

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict, post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        
        # Use the class with highest probability.
        pred = out.argmax(dim=1)  
        
        # Cache the predictions for calculating metrics
        predictions += list([x.item() for x in pred])
        true_labels += list([x.item() for x in data.label])
        
    
    return f1_score(true_labels, predictions, average='weighted')

"""
M
A
I
N
"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")

    wandb.login()
    
    sweep_configuration = sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'validation/weighted-f1'
        },
        'parameters': {
            'batch_size': {'values': [32, 64, 128, 256]},
            'epochs': {'max': 100, 'min': 5},
            'initial_lr': {'max': 0.015, 'min': 0.0001},
            'num_layers': {'values': [1,2,3]},
            'hidden_channels': {'values': [32, 64, 128, 256]},
            'dropout': {'max': 0.9, 'min': 0.2}
        }
    }
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, function=train, count=100)
        


