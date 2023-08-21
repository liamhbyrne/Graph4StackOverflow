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

log = setup_custom_logger("trainer", logging.INFO)


class GraphTrainer:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.log = setup_custom_logger("trainer", logging.INFO)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Proceeding with {self.device} . .")

        """
        Set linux multiprocessing strategy to avoid errors.
        """
        if self.config["OS_NAME"] == "linux":
            torch.multiprocessing.set_sharing_strategy("file_system")
            import resource
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

        if self.config["USE_WANDB"]:
            wandb_run_name = (
                f"run@{time.strftime('%Y%m%d-%H%M%S')}"
                if self.config["WANDB_RUN_NAME"] is None
                else self.config["WANDB_RUN_NAME"]
            )
            self.wandb_config = start_wandb_for_training(
                self.config["WANDB_PROJECT_NAME"], wandb_run_name
            )

            """
            Log hyperparameters to wandb
            """
            self.wandb_config.hidden_channels = self.config["HIDDEN_CHANNELS"]
            self.wandb_config.dropout = self.config["DROPOUT"]
            self.wandb_config.epoch = self.config["EPOCHS"]
            self.wandb_config.resampling = self.config["USE_CLASS_WEIGHTS_SAMPLER"]
            self.wandb_config.class_weights = self.config["USE_CLASS_WEIGHTS_LOSS"]
            self.wandb_config.include_answer = self.config["INCLUDE_ANSWER"]
            self.wandb_config.scheduler = "EXP"
            self.wandb_config.initial_lr = self.config["START_LR"]
            self.wandb_config.gamma = self.config["GAMMA"]
            self.wandb_config.batch_size = self.config["TRAIN_BATCH_SIZE"]

        """
        Model selection
        """
        if self.config["MODEL"] == "GAT":
            self._model = HeteroGAT(
                hidden_channels=self.config["HIDDEN_CHANNELS"], out_channels=2, num_layers=self.config["NUM_LAYERS"]
            )
        elif self.config["MODEL"] == "SAGE":
            self._model = HeteroGraphSAGE(
                hidden_channels=self.config["HIDDEN_CHANNELS"], out_channels=2, num_layers=self.config["HIDDEN_CHANNELS"]
            )
        elif self.config["MODEL"] == "GC":
            self._model = HeteroGraphConv(
                hidden_channels=self.config["HIDDEN_CHANNELS"], out_channels=2, num_layers=self.config["HIDDEN_CHANNELS"]
            )
        else:
            log.error(f"Model does not exist! ({self.config['MODEL']})")
            exit(1)

        self._model.to(self.device)

    def set_dataset(self):
        """
        Set train and test datasets
        """
        # Datasets
        if self.config["USE_IN_MEMORY_DATASET"]:
            self._train_dataset = UserGraphDatasetInMemory(
                root=self.config["ROOT"], file_name_out=self.config["TRAIN_DATA_PATH"], question_ids=[]
            )
            self._test_dataset = UserGraphDatasetInMemory(
                root=self.config["ROOT"], file_name_out=self.config["TEST_DATA_PATH"], question_ids=[]
            )
        else:
            dataset = UserGraphDataset(root=self.config["ROOT"], skip_processing=True)
            self._train_dataset, self._test_dataset = split_test_train_pytorch(dataset, 0.7)

        # (Optional) Take subset of training set
        if self.config["REL_SUBSET"] is not None:
            indices = list(range(int(len(self._train_dataset) * self.config["REL_SUBSET"])))
            self._train_dataset = torch.utils.data.Subset(self._train_dataset, indices)
            log.info(f"Subset contains {len(self._train_dataset)}")

    def get_dataloaders(self):
        """

        """
        train_loader = DataLoader(
            self._train_dataset,
            sampler=self._sampler,
            batch_size=self.config["TRAIN_BATCH_SIZE"],
            num_workers=self.config["NUM_WORKERS"],
        )

        test_loader = DataLoader(
            self._test_dataset,
            batch_size=self.config["TEST_BATCH_SIZE"],
            num_workers=self.config["NUM_WORKERS"],
        )
        return train_loader, test_loader


    def compile(self):
        """
        Set optimizer and scheduler
        """
        if self.config["WARM_START_FILE"] is not None:
            self._model.load_state_dict(torch.load(self.config["WARM_START_FILE"]), strict=False)

        # Optimizers & Schedulers
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config["START_LR"])
        self._scheduler = ExponentialLR(self._optimizer, gamma=self.config["GAMMA"], verbose=True)

        self._sampler = None
        class_weights = calculate_class_weights(self._train_dataset).to(self.device)

        """
        Set loss function
        """
        # (Optional) Sample by class weight
        if self.config["USE_CLASS_WEIGHTS_SAMPLER"]:
            train_labels = [x.label for x in self._train_dataset]
            self._sampler = torch.utils.data.WeightedRandomSampler(
                [class_weights[x] for x in train_labels], len(train_labels)
            )

        # Cross Entropy Loss (with optional class weights)
        self._criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights if self.config["USE_CLASS_WEIGHTS_LOSS"] else None
        )




    def train(self):
        """
        Set dataloaders
        """
        # Dataloaders
        log.info(f"Train DataLoader batch size is set to {self.config['TRAIN_BATCH_SIZE']}")

        train_loader, test_loader = self.get_dataloaders()

        self._model.train()

        for epoch in range(1, self.config["EPOCHS"] + 1):
            log.info(f"Epoch: {epoch:03d} > > >")

            # train model . .
            self.train_one_epoch(self._model, train_loader)

            # evaluate on training set . .
            train_info = self.test(train_loader)

            # evaluate on test set . .
            test_info = self.test(test_loader)

            # log for current epoch
            log.info(
                f'Epoch: {epoch:03d}, Test F1 (weighted): {test_info["f1-score-weighted"]:.4f}, Test F1 (macro): {test_info["f1-score-macro"]:.4f}'
            )

            # step scheduler
            self._scheduler.step()

            # print confusion matrix
            df = pd.DataFrame(
                {"actual": test_info["trues"], "prediction": test_info["preds"]}
            )
            confusion_matrix = pd.crosstab(
                df["actual"], df["prediction"], rownames=["Actual"], colnames=["Predicted"]
            )
            print(confusion_matrix)

            if self.config["SAVE_CHECKPOINTS"]:
                torch.save(self._model.state_dict(), f"../models/model-{epoch}.pt")

            # log evaluation results to wandb
            if self.config["USE_WANDB"]:
                log_results_to_wandb(train_info, "train")
                log_results_to_wandb(test_info, "test")

        log.info(f'Test F1: {test_info["f1-score-weighted"]:.4f}')

        save_model(self._model, self.config["FINAL_MODEL_OUT_PATH"])

        # Plot confusion matrix
        if self.config["USE_WANDB"]:
            add_cm_to_wandb(test_info)
            wandb.finish()


    def train_one_epoch(self, model, train_loader):
        """

        """
        for i, data in enumerate(
                train_loader
        ):  # Iterate in batches over the training dataset.
            data.to(self.device)

            self._optimizer.zero_grad()  # Clear gradients.

            if self.config["INCLUDE_ANSWER"]:
                # Concatenate question and answer embeddings to form post embeddings
                post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(self.device)
            else:
                # Use only question embeddings as post embedding
                post_emb = data.question_emb.to(self.device)
            post_emb.requires_grad = True

            out = self._model(
                data.x_dict, data.edge_index_dict, data.batch_dict, post_emb
            )  # Perform a single forward pass.

            # y = torch.tensor([1 if x > 0 else 0 for x in data.score]).to(device)
            loss = self._criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
            loss.backward()  # Derive gradients.
            self._optimizer.step()  # Update parameters based on gradients.

    def test(self):
        """

        """
        table = wandb.Table(columns=["ground_truth", "prediction"]) if self.config["USE_WANDB"] else None
        self._model.eval()

        predictions = []
        true_labels = []

        cumulative_loss = 0

        test_loader = DataLoader(
            self._test_dataset, batch_size=self.config["TEST_BATCH_SIZE"], num_workers=self.config["NUM_WORKERS"]
        )

        for data in test_loader:  # Iterate in batches over the training/test dataset.
            data.to(self.device)

            if self.config["INCLUDE_ANSWER"]:
                post_emb = torch.cat([data.question_emb, data.answer_emb], dim=1).to(
                    self.device
                )
            else:
                post_emb = data.question_emb.to(self.device)

            out = self._model(
                data.x_dict, data.edge_index_dict, data.batch_dict, post_emb
            )  # Perform a single forward pass.

            # y = torch.tensor([1 if x > 0 else 0 for x in data.score]).to(device)
            loss = self._criterion(out, torch.squeeze(data.label, -1))  # Compute the loss.
            cumulative_loss += loss.item()

            # Use the class with the highest probability.
            pred = out.argmax(dim=1)

            # Cache the predictions for calculating metrics
            predictions += list([x.item() for x in pred])
            true_labels += list([x.item() for x in data.label])

            # Log table of predictions to WandB
            if self.config["USE_WANDB"]:
                # graph_html = wandb.Html(plotly.io.to_html(create_graph_vis(data)))

                for pred, label in zip(pred, torch.squeeze(data.label, -1)):
                    table.add_data(label, pred)

        # Collate results into a single dictionary
        test_results = {
            "accuracy": accuracy_score(true_labels, predictions),
            "f1-score-weighted": f1_score(true_labels, predictions, average="weighted"),
            "f1-score-macro": f1_score(true_labels, predictions, average="macro"),
            "loss": cumulative_loss / len(test_loader),
            "table": table,
            "preds": predictions,
            "trues": true_labels,
        }
        return test_results

    def k_fold_cross_validation(self):
        """

        """
        pass


    def main(self):
        self.log.info(f"Proceeding with {self.device} . .")

        # Rest of your main logic here
        self.set_dataset()
        self.compile()
        self.train()


if __name__ == "__main__":
    trainer = GraphTrainer("model_config.yaml")
    trainer.main()
