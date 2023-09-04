import logging
import pickle
import time

import pandas as pd
import torch
import wandb
import yaml

from ACL2024.modules.models.GNNs.hetero_GAT import HeteroGAT
from ACL2024.modules.models.GNNs.hetero_GraphConv import HeteroGraphConv
from ACL2024.modules.models.GNNs.hetero_GraphSAGE import HeteroGraphSAGE
from ACL2024.modules.models.helper_functions import calculate_class_weights, split_test_train_pytorch, start_wandb_for_training, log_results_to_wandb, save_model, add_cm_to_wandb
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.loader import DataLoader

from ACL2024.modules.dataset.compile_dataset import UserGraphDatasetInMemory
from ACL2024.modules.dataset.user_graph_dataset import UserGraphDataset
from ACL2024.modules.util.custom_logger import setup_custom_logger


with open("model_config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)


log = setup_custom_logger("trainer", logging.INFO)

if CONFIG["OS_NAME"] == "linux":
    torch.multiprocessing.set_sharing_strategy("file_system")
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def train(model, train_loader):
    running_loss = 0.0
    model.train()

    for i, data in enumerate(
        train_loader
    ):  # Iterate in batches over the training dataset.
        data.to(device)

        optimizer.zero_grad()  # Clear gradients.

        if CONFIG["INCLUDE_ANSWER"]:
            # Concatenate question and answer embeddings to form post embeddings
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            # Use only question embeddings as post embedding
            post_emb = data.question_emb.to(device)
        post_emb.requires_grad = True

        out = model(
            data.x_dict, data.edge_index_dict, data.batch_dict, post_emb
        )  # Perform a single forward pass.

        # y = torch.tensor([1 if x > 0 else 0 for x in data.score]).to(device)
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
    table = wandb.Table(columns=["ground_truth", "prediction"]) if CONFIG["USE_WANDB"] else None
    model.eval()

    predictions = []
    true_labels = []

    cumulative_loss = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)

        if CONFIG["INCLUDE_ANSWER"]:
            post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(device)
        else:
            post_emb = data.question_emb.to(device)

        out = model(
            data.x_dict, data.edge_index_dict, data.batch_dict, post_emb
        )  # Perform a single forward pass.

        # y = torch.tensor([1 if x > 0 else 0 for x in data.score]).to(device)
        loss = criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
        cumulative_loss += loss.item()

        # Use the class with highest probability.
        pred = out.argmax(dim=1)

        # Cache the predictions for calculating metrics
        predictions += list([x.item() for x in pred])
        true_labels += list([x.item() for x in data.label])

        # Log table of predictions to WandB
        if CONFIG["USE_WANDB"]:
            # graph_html = wandb.Html(plotly.io.to_html(create_graph_vis(data)))

            for pred, label in zip(pred, torch.squeeze(data.label, -1)):
                table.add_data(label, pred)

    # Collate results into a single dictionary
    test_results = {
        "accuracy": accuracy_score(true_labels, predictions),
        "f1-score-weighted": f1_score(true_labels, predictions, average="weighted"),
        "f1-score-macro": f1_score(true_labels, predictions, average="macro"),
        "loss": cumulative_loss / len(loader),
        "table": table,
        "preds": predictions,
        "trues": true_labels,
    }
    return test_results


"""
M
A
I
N
"""
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Proceeding with {device} . .")

    if CONFIG["USE_WANDB"]:
        log.info(f"Connecting to Weights & Biases . .")
        if CONFIG["WANDB_RUN_NAME"] is None:
            WANDB_RUN_NAME = f"run@{time.strftime('%Y%m%d-%H%M%S')}"
        config = start_wandb_for_training(
            CONFIG["WANDB_PROJECT_NAME"], CONFIG["WANDB_RUN_NAME"]
        )
        config.hidden_channels = CONFIG["HIDDEN_CHANNELS"]
        config.dropout = CONFIG["DROPOUT"]
        config.epoch = CONFIG["EPOCHS"]
        config.resampling = CONFIG["USE_CLASS_WEIGHTS_SAMPLER"]
        config.class_weights = CONFIG["USE_CLASS_WEIGHTS_LOSS"]
        config.include_answer = CONFIG["INCLUDE_ANSWER"]
        config.scheduler = "EXP"
        config.initial_lr = CONFIG["START_LR"]
        config.gamma = CONFIG["GAMMA"]
        config.batch_size = CONFIG["TRAIN_BATCH_SIZE"]

    # Datasets
    if CONFIG["USE_IN_MEMORY_DATASET"]:
        train_dataset = UserGraphDatasetInMemory(
            root=CONFIG["ROOT"], file_name_out=CONFIG["TRAIN_DATA_PATH"], question_ids=[]
        )
        test_dataset = UserGraphDatasetInMemory(
            root=CONFIG["ROOT"], file_name_out=CONFIG["TEST_DATA_PATH"], question_ids=[]
        )
    else:
        dataset = UserGraphDataset(root=CONFIG["ROOT"], skip_processing=True)
        train_dataset, test_dataset = split_test_train_pytorch(dataset, 0.7)

    if CONFIG["USE_KFOLD"]:

        folds = [
            UserGraphDatasetInMemory(
                root="../data", file_name_out=fold_path, question_ids=[]
            )
            for fold_path in CONFIG["FOLD_FILES"]
        ]

        kfold_results = []

        for i in range(len(folds)):
            test_fold = folds[i]
            train_fold = torch.utils.data.ConcatDataset(
                [fold for j, fold in enumerate(folds) if j != i]
            )

            log.info(f"Fold {i + 1} of {len(folds)}\n-------------------")

            sampler = None
            class_weights = calculate_class_weights(train_fold).to(device)

            # Sample by class weight
            if CONFIG["USE_CLASS_WEIGHTS_SAMPLER"]:
                train_labels = [x.label for x in train_fold]
                sampler = torch.utils.data.WeightedRandomSampler(
                    [class_weights[x] for x in train_labels], len(train_labels)
                )

            # Define data loaders for training and testing data in this fold
            train_fold_loader = DataLoader(
                train_fold, batch_size=CONFIG["TRAIN_BATH_SIZE"], sampler=sampler
            )
            test_fold_loader = DataLoader(test_fold, batch_size=CONFIG["TEST_BATCH_SIZE"])
            # Model
            model = HeteroGAT(
                hidden_channels=CONFIG["HIDDEN_CHANNELS"], out_channels=2, num_layers=CONFIG["NUM_LAYERS"]
            )
            model.to(device)  # To GPU if available

            # Optimizers & Loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["START_LR"])
            scheduler = ExponentialLR(optimizer, gamma=CONFIG["GAMMA"], verbose=True)

            # Optional class weights on the criterion
            fold_class_weights = calculate_class_weights(train_fold).to(device)
            criterion = torch.nn.CrossEntropyLoss(
                weight=fold_class_weights if CONFIG["USE_CLASS_WEIGHTS_LOSS"] else None
            )
            for epoch in range(1, CONFIG["EPOCHS"] + 1):
                log.info(f"Epoch: {epoch:03d} > > >")

                # train model . .
                train(model, train_fold_loader)

                # evaluate on test fold . .
                test_info = test(test_fold_loader)

                # log for current epoch
                log.info(
                    f'Epoch: {epoch:03d}, Test F1 (weighted): {test_info["f1-score-weighted"]:.4f}, Test F1 (macro): {test_info["f1-score-macro"]:.4f}'
                )

                # step scheduler
                scheduler.step()

                # print confusion matrix
                df = pd.DataFrame(
                    {"actual": test_info["trues"], "prediction": test_info["preds"]}
                )
                confusion_matrix = pd.crosstab(
                    df["actual"],
                    df["prediction"],
                    rownames=["Actual"],
                    colnames=["Predicted"],
                )
                print(confusion_matrix)

            log.info(f'Fold {i+1}, Test F1: {test_info["f1-score"]:.4f}')
            kfold_results.append(test_info)

        print(f"K-Fold Results: {kfold_results}")
        # Pickle results
        if CONFIG["PICKLE_PATH_KF"] is not None:
            with open(CONFIG["PICKLE_PATH_KF"], "wb") as f:
                pickle.dump(kfold_results, f)

    log.info(f"Sample graph:\n{train_dataset[0]}")
    log.info(f"Train Dataset Size: {len(train_dataset)}")
    log.info(f"Test Dataset Size: {len(test_dataset)}")

    # Weights&Biases dashboard
    data_details = {
        "num_node_features": train_dataset.num_node_features,
        "num_classes": 2,
    }
    log.info(f"Data Details:\n{data_details}")
    if CONFIG["USE_WANDB"]:
        wandb.log(data_details)

    # Take subset for EXP3
    if CONFIG["REL_SUBSET"] is not None:
        indices = list(range(int(len(train_dataset) * CONFIG["REL_SUBSET"])))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        log.info(f"Subset contains {len(train_dataset)}")

    sampler = None
    class_weights = calculate_class_weights(train_dataset).to(device)

    # Sample by class weight
    if CONFIG["USE_CLASS_WEIGHTS_SAMPLER"]:
        train_labels = [x.label for x in train_dataset]
        sampler = torch.utils.data.WeightedRandomSampler(
            [class_weights[x] for x in train_labels], len(train_labels)
        )

    # Dataloaders
    log.info(f"Train DataLoader batch size is set to {CONFIG['TRAIN_BATCH_SIZE']}")
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=CONFIG["TRAIN_BATCH_SIZE"],
        num_workers=CONFIG["NUM_WORKERS"],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["TEST_BATCH_SIZE"], num_workers=CONFIG["NUM_WORKERS"]
    )

    # Model
    if CONFIG["MODEL"] == "GAT":
        model = HeteroGAT(
            hidden_channels=CONFIG["HIDDEN_CHANNELS"], out_channels=2, num_layers=CONFIG["NUM_LAYERS"]
        )
    elif CONFIG["MODEL"] == "SAGE":
        model = HeteroGraphSAGE(
            hidden_channels=CONFIG["HIDDEN_CHANNELS"], out_channels=2, num_layers=CONFIG["NUM_LAYERS"]
        )
    elif CONFIG["MODEL"] == "GC":
        model = HeteroGraphConv(
            hidden_channels=CONFIG["HIDDEN_CHANNELS"], out_channels=2, num_layers=CONFIG["NUM_LAYERS"]
        )
    else:
        log.error(f"Model does not exist! ({CONFIG['MODEL']})")
        exit(1)

    """
    Put model on GPU if available
    """


    if CONFIG["WARM_START_FILE"] is not None:
        model.load_state_dict(torch.load(CONFIG["WARM_START_FILE"]), strict=False)

    # Optimizers & Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["START_LR"])
    scheduler = ExponentialLR(optimizer, gamma=CONFIG["GAMMA"], verbose=True)

    # Cross Entropy Loss (with optional class weights)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights if CONFIG["USE_CLASS_WEIGHTS_LOSS"] else None
    )

    for epoch in range(1, CONFIG["EPOCH"] + 1):
        log.info(f"Epoch: {epoch:03d} > > >")

        # train model . .
        train(model, train_loader)

        # evaluate on training set . .
        train_info = test(train_loader)

        # evaluate on test set . .
        test_info = test(test_loader)

        # log for current epoch
        log.info(
            f'Epoch: {epoch:03d}, Test F1 (weighted): {test_info["f1-score-weighted"]:.4f}, Test F1 (macro): {test_info["f1-score-macro"]:.4f}'
        )

        # step scheduler
        scheduler.step()

        # print confusion matrix
        df = pd.DataFrame(
            {"actual": test_info["trues"], "prediction": test_info["preds"]}
        )
        confusion_matrix = pd.crosstab(
            df["actual"], df["prediction"], rownames=["Actual"], colnames=["Predicted"]
        )
        print(confusion_matrix)

        if CONFIG["SAVE_CHECKPOINTS"]:
            torch.save(model.state_dict(), f"../models/model-{epoch}.pt")

        # log evaluation results to wandb
        if CONFIG["USE_WANDB"]:
            log_results_to_wandb(train_info, "train")
            log_results_to_wandb(test_info, "test")

    log.info(f'Test F1: {test_info["f1-score-weighted"]:.4f}')

    save_model(model, CONFIG["FINAL_MODEL_OUT_PATH"])
    # Plot confusion matrix
    if CONFIG["USE_WANDB"]:
        add_cm_to_wandb(test_info)
        wandb.finish()
