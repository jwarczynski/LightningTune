import os
import time
import argparse
import random

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import optuna
from optuna.exceptions import DuplicatedStudyError

import numpy as np

import wandb

from models import ConvNet


def set_seed(seed: int):
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch (CPU)
    torch.manual_seed(seed)

    # Avoid nondeterministic algorithms if possible
    torch.use_deterministic_algorithms(True)

    # Set seed for PyTorch (GPU) - if you're using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MNISTData(L.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        train_set = datasets.FashionMNIST("data", train=True, download=True,
                                          transform=transforms.Compose([transforms.ToTensor()]))
        self.test_set = datasets.FashionMNIST("data", train=False, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
        self.train_set, self.val_set = torch.utils.data.random_split(train_set, [50000, 10000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=11)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=11)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


class Classifier(L.LightningModule):
    def __init__(self, lr=0.001, dropout=0.25, weight_decay=1e-5):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = ConvNet(dropout)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/accuracy": accuracy})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    model = Classifier(lr=lr, dropout=dropout, weight_decay=weight_decay)

    wandb_logger = WandbLogger(
        project=args.project_name, tags=[f"lr={lr}", f"dropout={dropout}", f"weight_decay={weight_decay}"],
        name=f"lr={lr}_dropout={dropout}_weight_decay={weight_decay}")
    wandb_logger.watch(model, log="all")

    data = MNISTData(64)
    trainer = L.Trainer(max_epochs=10, logger=wandb_logger,
                        limit_val_batches=args.limit_val_batches,
                        limit_train_batches=args.limit_train_batches,
                        enable_model_summary=False)

    trainer.fit(model, data)

    # enable logs for next runs
    wandb.finish()
    return trainer.logged_metrics["val/accuracy"]


def load_or_create_study(study_name, storage_url, direction="maximize", max_retries=5, retry_interval=5):
    """Attempt to load an Optuna study, creating it if it doesn't exist.
    Retries in case of concurrent access by other workers.
    """
    for attempt in range(max_retries):
        try:
            # Check if the study already exists
            study_summaries = optuna.study.get_all_study_summaries(storage=storage_url)
            if any(summary.study_name == study_name for summary in study_summaries):
                # Study already exists; load it
                print(f"Loading existing study '{study_name}'")
                return optuna.load_study(study_name=study_name, storage=storage_url)

            # Study does not exist; attempt to create it
            print(f"Creating new study '{study_name}'")
            return optuna.create_study(study_name=study_name, storage=storage_url, direction=direction)

        except DuplicatedStudyError:
            # Handle concurrent study creation by another worker
            print(f"Study '{study_name}' already exists (created by another worker), retrying load.")
            time.sleep(retry_interval)

        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            time.sleep(retry_interval)

    raise RuntimeError(f"Failed to load or create study '{study_name}' after {max_retries} attempts")


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_trials", type=int, default=5)
parser.add_argument("-ltb", "--limit_train_batches", type=float, default=0.2)
parser.add_argument("-lvb", "--limit_val_batches", type=float, default=0.2)
parser.add_argument("-sn", "--study_name", type=str, required=True)
parser.add_argument("-pn", "--project_name", type=str, required=True)

args = parser.parse_args()


if __name__ == "__main__":
    set_seed(44)
    wandb.login(anonymous="never", key=os.environ["WANDB_API_KEY"])

    # Load or create the study
    study = load_or_create_study(
        study_name=args.study_name,
        storage_url=os.getenv("OPTUNA_DB_URL"),
        direction="maximize"
    )

    # Run the optimization
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=True)
