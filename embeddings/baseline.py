import logging
import logging
import time

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear

import helper_functions
import wandb
from baseline_constants import OS_NAME, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, IN_MEMORY_DATASET, INCLUDE_ANSWER, USE_WANDB, WANDB_PROJECT_NAME, NUM_WORKERS, EPOCHS, HIDDEN_CHANNELS, FINAL_MODEL_OUT_PATH, SAVE_CHECKPOINTS, WANDB_RUN_NAME, CROSS_VALIDATE, FOLD_FILES, USE_CLASS_WEIGHTS_SAMPLER, USE_CLASS_WEIGHTS_LOSS, DROPOUT
from custom_logger import setup_custom_logger
from dataset import UserGraphDataset
from dataset_in_memory import UserGraphDatasetInMemory
from helper_functions import calculate_class_weights, split_test_train_pytorch

log = setup_custom_logger("baseline_model", logging.INFO)

if OS_NAME == "linux":
    torch.multiprocessing.set_sharing_strategy('file_system')
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


class Baseline(torch.nn.Module):
    """
    Heterogeneous Graph Attentional Network (GAT) model.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(-1, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, post_emb):
        post_emb = torch.squeeze(post_emb)
        out = self.lin1(post_emb).relu()
        out = F.dropout(out, p=DROPOUT, training=self.training)

        out = self.lin2(out).relu()
        #out = F.dropout(post_emb, p=DROPOUT, training=self.training)

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
            # Concatenate question and answer embeddings to form post embeddings
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            # Use only question embeddings as post embedding
            post_emb = data.question_emb.to(device)
        post_emb.requires_grad = True

        out = model(post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()

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

        out = model(post_emb)  # Perform a single forward pass.

        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        cumulative_loss += loss.item()

        # Use the class with highest probability.
        pred = out.argmax(dim=1)

        # Cache the predictions for calculating metrics
        predictions += list([x.item() for x in pred])
        true_labels += list([x.item() for x in data.label])

        # Log table of predictions to WandB
        if USE_WANDB:
            # graph_html = wandb.Html(plotly.io.to_html(create_graph_vis(data)))

            for pred, label in zip(pred, torch.squeeze(data.label, -1)):
                table.add_data(label, pred)

    # Collate results into a single dictionary
    test_results = {
        "accuracy": accuracy_score(true_labels, predictions),
        "f1-score": f1_score(true_labels, predictions, average='binary'),
        "loss": cumulative_loss / len(loader),
        "table": table,
        "preds": predictions,
        "trues": true_labels
    }
    return test_results


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")

    if USE_WANDB:
        log.info(f"Connecting to Weights & Biases . .")
        if WANDB_RUN_NAME is None:
            WANDB_RUN_NAME = f"baseline-run@{time.strftime('%Y%m%d-%H%M%S')}"
        helper_functions.start_wandb_for_training(WANDB_PROJECT_NAME, WANDB_RUN_NAME)

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
            # Define data loaders for training and testing data in this fold
            train_fold_loader = DataLoader(
                train_fold,
                batch_size=TRAIN_BATCH_SIZE
            )
            test_fold_loader = DataLoader(
                test_fold,
                batch_size=TEST_BATCH_SIZE
            )
            # Model
            model = Baseline(out_channels=2, hidden_channels=HIDDEN_CHANNELS)
            model.to(device)  # To GPU if available

            # Optimizers & Loss function
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

            fold_class_weights = calculate_class_weights(train_fold).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=fold_class_weights if USE_CLASS_WEIGHTS_LOSS else None)
            for epoch in range(1, EPOCHS):
                log.info(f"Epoch: {epoch:03d} > > >")

                # train model . .
                train(model, train_fold_loader)

                # evaluate on test fold . .
                test_info = test(test_fold_loader)

                # log for current epoch
                log.info(f'Epoch: {epoch:03d}, Test F1: {test_info["f1-score"]:.4f}')

            log.info(f'Fold {i + 1}, Test F1: {test_info["f1-score"]:.4f}')
            kfold_results.append(test_info)

        print(f"K-Fold Results: {kfold_results}")

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
    model = Baseline(out_channels=2, hidden_channels=HIDDEN_CHANNELS)
    model.to(device)  # To GPU if available

    # Optimizers & Loss function
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Cross Entropy Loss (with optional class weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights if USE_CLASS_WEIGHTS_LOSS else None)

    for epoch in range(1, EPOCHS + 1):
        log.info(f"Epoch: {epoch:03d} > > >")

        # train model . .
        train(model, train_loader)

        # evaluate on training set . .
        train_info = test(train_loader)

        # evaluate on test set . .
        test_info = test(test_loader)

        # log for current epoch
        log.info(f'Epoch: {epoch:03d}, Loss {train_info["loss"]}, Train F1: {train_info["f1-score"]:.4f}, Test F1: {test_info["f1-score"]:.4f}')

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
