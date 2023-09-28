import os

import luigi
import numpy as np
import torch
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from utils.models.deepjet import DeepJet
from utils.torch.datasets import DeepJetDataset
from utils.torch.training import perform_training
from utils.config.config_loader import ConfigLoader

torch.autograd.detect_anomaly(True)


class TrainingTask(TrainingDependency, DatasetDependency, BaseTask):
    loss_weighting = luigi.BoolParameter(
        False,
        description="Whether to weight the loss or use weighted sampling from the dataset",
    )

    n_threads = luigi.IntParameter(
        default=4, description="Number of threads to use for dataloader."
    )

    def requires(self):
        return DatasetConstructorTask.req(self)

    def output(self):
        return {
            "training_metrics": self.local_target("training_metrics.npz"),
            "validation_metrics": self.local_target("validation_metrics.npz"),
            "model": self.local_target(f"model_{self.epochs-1}.pt"),
            "best_model": self.local_target("best_model.pt"),
        }

    def run(self):
        # Loading config
        config_dict = np.load(self.input()["config_dict"].path, allow_pickle=True).item()
        config = ConfigLoader.load_config(self.config)
        os.makedirs(self.local_path(), exist_ok=True)
        print("Loading Dataset")
        files = np.array(self.input()["file_list"].load().split("\n"))

        training_mask = ~(np.char.find(files, "train") == -1)
        validation_mask = ~(np.char.find(files, "validation") == -1)

        training_files = files[training_mask]
        validation_files = files[validation_mask]

        histogram_training = np.load(
            self.input()["histogram_training"].path,
            allow_pickle=True,
        )

        batch_size = 10000

        # Define the training and validation datasets
        training_data = DeepJetDataset(
            training_files,
            "training",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            histogram_training=histogram_training,
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
        )
        validation_data = DeepJetDataset(
            validation_files,
            "validation",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            histogram_training=histogram_training,
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
        )

        # Define the corresponding dataloaders
        training_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=True,  # Pin Memory for faster CPU/GPU memory load
            num_workers=self.n_threads,
        )
        # Expected number of iterations
        training_dataloader.nits_expected = len(training_dataloader)

        validation_dataloader = DataLoader(
            validation_data,
            batch_size=batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.n_threads,
        )
        validation_dataloader.nits_expected = len(validation_dataloader)

        # Model Defintion
        print("Model definition")
        model = DeepJet(config_dict["model"]["feature_edges"]).to(self.device)
        # model = torch.compile(model)
        scaler = torch.cuda.amp.GradScaler()

        # Training
        print("Start training on " + self.device)
        train_metrics, validation_metrics = perform_training(
            model,
            training_dataloader,
            validation_dataloader,
            self.local_path(),
            self.device,
            nepochs=self.epochs,
        )

        print("Training finished. Saving data...")

        np.savez(
            self.output()["training_metrics"].path,
            loss=train_metrics[:, 0],
            acc=train_metrics[:, 1],
            allow_pickle=True,
        )
        np.savez(
            self.output()["validation_metrics"].path,
            loss=validation_metrics[:, 0],
            acc=validation_metrics[:, 1],
            allow_pickle=True,
        )
