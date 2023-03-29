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
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, global_mean_pool, GCNConv, SAGEConv
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
from hetero_GAT_constants import OS_NAME, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, IN_MEMORY_DATASET, INCLUDE_ANSWER, USE_WANDB, WANDB_PROJECT_NAME, NUM_WORKERS, EPOCHS, NUM_LAYERS, HIDDEN_CHANNELS, FINAL_MODEL_OUT_PATH, SAVE_CHECKPOINTS, WANDB_RUN_NAME, CROSS_VALIDATE, FOLD_FILES, USE_CLASS_WEIGHTS_SAMPLER, USE_CLASS_WEIGHTS_LOSS, DROPOUT, GAMMA, START_LR, PICKLE_PATH_KF

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

        self.convs = torch.nn.ModuleList()

        # Create Graph Attentional layers
        for _ in range(num_layers):
            conv = HeteroConv({
                ('tag', 'describes', 'question'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('tag', 'describes', 'answer'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('tag', 'describes', 'comment'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('module', 'imported_in', 'question'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('module', 'imported_in', 'answer'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('question', 'rev_describes', 'tag'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('answer', 'rev_describes', 'tag'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('comment', 'rev_describes', 'tag'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('question', 'rev_imported_in', 'module'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
                ('answer', 'rev_imported_in', 'module'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=6),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        for conv in self.convs:
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
G
C
N
"""

class HeteroGCN(torch.nn.Module):
    """
    Heterogeneous Graph Convolutional Network (GCN) model.
    """
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        # Create Graph Attentional layers
        for _ in range(num_layers):
            conv = HeteroConv({
                ('tag', 'describes', 'question'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'answer'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'comment'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'question'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'answer'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('question', 'rev_describes', 'tag'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_describes', 'tag'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('comment', 'rev_describes', 'tag'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('question', 'rev_imported_in', 'module'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_imported_in', 'module'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        for conv in self.convs:
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
G
R
A
P
H
S
A
G
E
"""

class HeteroGraphSAGE(torch.nn.Module):
    """
    Heterogeneous GraphSAGE model.
    """
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        # Create Graph Attentional layers
        for _ in range(num_layers):
            conv = HeteroConv({
                ('tag', 'describes', 'question'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'answer'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('tag', 'describes', 'comment'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'question'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('module', 'imported_in', 'answer'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('question', 'rev_describes', 'tag'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_describes', 'tag'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('comment', 'rev_describes', 'tag'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('question', 'rev_imported_in', 'module'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                ('answer', 'rev_imported_in', 'module'): GCNConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        for conv in self.convs:
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
def train(model, train_loader):
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
M
A
I
N
"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")
    
    if USE_WANDB:
        log.info(f"Connecting to Weights & Biases . .")
        if WANDB_RUN_NAME is None:
            WANDB_RUN_NAME = f"run@{time.strftime('%Y%m%d-%H%M%S')}"
        config = helper_functions.start_wandb_for_training(WANDB_PROJECT_NAME, WANDB_RUN_NAME)
        config.hidden_channels = HIDDEN_CHANNELS
        config.dropout = DROPOUT
        config.epoch = EPOCHS
        config.resampling = USE_CLASS_WEIGHTS_SAMPLER
        config.class_weights = USE_CLASS_WEIGHTS_LOSS
        config.include_answer = INCLUDE_ANSWER
        config.scheduler = "EXP"
        config.initial_lr = START_LR
        config.gamma = GAMMA
        config.batch_size = TRAIN_BATCH_SIZE
    

    # Datasets
    if IN_MEMORY_DATASET:
        train_dataset = UserGraphDatasetInMemory(root="../data", file_name_out='train-4175-qs.pt', question_ids=[])
        test_dataset = UserGraphDatasetInMemory(root="../data", file_name_out='test-1790-qs.pt', question_ids=[])
    else:
        dataset = UserGraphDataset(root="../data", skip_processing=True)
        train_dataset, test_dataset = split_test_train_pytorch(dataset, 0.7)

    if CROSS_VALIDATE:
        print(FOLD_FILES)
        folds = [UserGraphDatasetInMemory(root="../data", file_name_out=fold_path, question_ids=[]) for fold_path in FOLD_FILES]
        kfold_results = []
        for i in range(len(folds)):
            test_fold = folds[i]
            train_fold = torch.utils.data.ConcatDataset([fold for j, fold in enumerate(folds) if j != i])

            log.info(f"Fold {i + 1} of {len(folds)}\n-------------------")
            
            sampler = None
            class_weights = calculate_class_weights(train_fold).to(device)
    
            # Sample by class weight
            if USE_CLASS_WEIGHTS_SAMPLER:
                train_labels = [x.label for x in train_fold]
                sampler = torch.utils.data.WeightedRandomSampler([class_weights[x] for x in train_labels], len(train_labels))
            
            # Define data loaders for training and testing data in this fold
            train_fold_loader = DataLoader(
                train_fold,
                batch_size=TRAIN_BATCH_SIZE,
                sampler=sampler
            )
            test_fold_loader = DataLoader(
                test_fold,
                batch_size=TEST_BATCH_SIZE
            )
            # Model
            model = HeteroGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=2, num_layers=NUM_LAYERS)
            model.to(device)  # To GPU if available

            # Optimizers & Loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)
            scheduler = ExponentialLR(optimizer, gamma=GAMMA, verbose=True)
            
            # Optional class weights on the criterion
            fold_class_weights = calculate_class_weights(train_fold).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=fold_class_weights if USE_CLASS_WEIGHTS_LOSS else None)
            for epoch in range(1, EPOCHS):
                log.info(f"Epoch: {epoch:03d} > > >")

                # train model . .
                train(model, train_fold_loader)

                # evaluate on test fold . .
                test_info = test(test_fold_loader)

                # log for current epoch
                log.info(f'Epoch: {epoch:03d}, Test F1 (weighted): {test_info["f1-score-weighted"]:.4f}, Test F1 (macro): {test_info["f1-score-macro"]:.4f}')
                
                # step scheduler
                scheduler.step()
                 
                # print confusion matrix
                df = pd.DataFrame({'actual': test_info["trues"], 'prediction': test_info["preds"]})
                confusion_matrix = pd.crosstab(df['actual'], df['prediction'], rownames=['Actual'], colnames=['Predicted'])
                print(confusion_matrix)


            log.info(f'Fold {i+1}, Test F1: {test_info["f1-score"]:.4f}')
            kfold_results.append(test_info)

        print(f"K-Fold Results: {kfold_results}")
        # Pickle results
        if PICKLE_PATH_KF is not None:
          with open(PICKLE_PATH_KF, "wb") as f:
              pickle.dump(kfold_results, f)
            


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
        wandb.log(data_details)

    sampler = None
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # Sample by class weight
    if USE_CLASS_WEIGHTS_SAMPLER:
        train_labels = [x.label for x in train_dataset]
        sampler = torch.utils.data.WeightedRandomSampler([class_weights[x] for x in train_labels], len(train_labels))

    # Dataloaders
    log.info(f"Train DataLoader batch size is set to {TRAIN_BATCH_SIZE}")
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS)

    # Model
    model = HeteroGraphSAGE(hidden_channels=HIDDEN_CHANNELS, out_channels=2, num_layers=NUM_LAYERS)
    model.to(device)  # To GPU if available
    
    # Optimizers & Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)
    scheduler = ExponentialLR(optimizer, gamma=GAMMA, verbose=True)
    
    # Cross Entropy Loss (with optional class weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights if USE_CLASS_WEIGHTS_LOSS else None)


    for epoch in range(1, EPOCHS+1):
        log.info(f"Epoch: {epoch:03d} > > >")
        
        # train model . . 
        train(model, train_loader)

        # evaluate on training set . . 
        train_info = test(train_loader)

        # evaluate on test set . . 
        test_info = test(test_loader)

        # log for current epoch
        log.info(f'Epoch: {epoch:03d}, Test F1 (weighted): {test_info["f1-score-weighted"]:.4f}, Test F1 (macro): {test_info["f1-score-macro"]:.4f}')
                
        # step scheduler
        scheduler.step()
        
        # print confusion matrix
        df = pd.DataFrame({'actual': test_info["trues"], 'prediction': test_info["preds"]})
        confusion_matrix = pd.crosstab(df['actual'], df['prediction'], rownames=['Actual'], colnames=['Predicted'])
        print(confusion_matrix)

        if SAVE_CHECKPOINTS:
            torch.save(model.state_dict(), f"../models/model-{epoch}.pt")
        
        # log evaluation results to wandb
        if USE_WANDB:
            helper_functions.log_results_to_wandb(train_info, "train")
            helper_functions.log_results_to_wandb(test_info, "test")

    log.info(f'Test F1: {test_info["f1-score"]:.4f}')

    helper_functions.save_model(model, FINAL_MODEL_OUT_PATH)
    # Plot confusion matrix
    if USE_WANDB:
        helper_functions.add_cm_to_wandb(test_info)
        wandb.finish()

















