import os

import luigi
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from utils.config.config_loader import ConfigLoader
from utils.models.models import BTaggingModels


def check_resume(base_path, model_prefix="model_", model_suffix=".pt", load_epoch=None):
    models = {}
    for p in Path(base_path).glob(f"{model_prefix}[0-9]*{model_suffix}"):
        path_name = str(p)
        name = path_name.split("/")[-1]
        epoch = int(name.replace(model_prefix, "").replace(model_suffix, ""))
        models[epoch] = path_name
    if len(models.values()) == 0:
        raise FileNotFoundError
    else:
        if not load_epoch:
            max_epoch = max(models)
            return models[max_epoch], max_epoch
        else:
            return models[load_epoch], load_epoch


def load_resume_training(model, path, device, epoch=None):
    try:
        model_path, ran_epochs = check_resume(path, load_epoch=epoch)
        _model = torch.load(
            model_path,
            map_location=torch.device(device),
        )
        model.load_state_dict(_model["model_state_dict"])
        print(f"Resuming on epoch {ran_epochs}:\n{model_path}")
        return model, ran_epochs
    except FileNotFoundError:
        print("No training to resume found. Starting a new one")
        return model, 0


class TrainingTask(TrainingDependency, DatasetDependency, BaseTask):
    loss_weighting = luigi.BoolParameter(
        False,
        description="Whether to weight the loss or use weighted sampling from the dataset",
    )

    resume_training = luigi.BoolParameter(
        False,
        description="Whether to resume the training if it already ran partially and failed. Set this to true if you want to resume.",
    )
    resume_epoch = luigi.IntParameter(
        False,
        description="Whether to resume the training from a specific epoch",
    )

    extend_training = luigi.IntParameter(
        0,
        description="Number of epochs to extend a training.",
    )

    train_val_split = 0.9

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
        config = ConfigLoader.load_config(self.config)
        os.makedirs(self.local_path(), exist_ok=True)
        print("Loading Dataset")
        files = np.array(self.input()["file_list"].load().split("\n"))

        n_train = int(len(files) * self.train_val_split)
        training_files = files[:n_train]
        validation_files = files[n_train:]
        print(f"#Train files: {n_train}")
        print(f"#Val files: {len(files) - n_train}")

        histogram_training = np.load(
            self.input()["histogram"].path,
            allow_pickle=True,
        )

        # Model Defintion
        model = BTaggingModels(self.model_name).to(self.device)
        print("Model construction")
        if self.resume_training or self.resume_epoch:
            model, ran_epochs = load_resume_training(
                model, self.local_path(), self.device, self.resume_epoch
            )
        else:
            ran_epochs = 0
        datasetClass = model.datasetClass
        # Define the training and validation datasets
        training_data = datasetClass(
            training_files,
            model=model,
            data_type="training",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            histogram_training=histogram_training,
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
            verbose=self.verbose,
        )
        validation_data = datasetClass(
            validation_files,
            model=model,
            data_type="validation",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            histogram_training=None,
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
            verbose=self.verbose,
        )

        # Define the corresponding dataloaders
        training_dataloader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,  # Pin Memory for faster CPU/GPU memory load
            num_workers=self.n_threads,
        )
        # Expected number of iterations
        training_dataloader.nits_expected = len(training_dataloader)

        validation_dataloader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.n_threads,
        )
        validation_dataloader.nits_expected = len(validation_dataloader)

        # Training
        print("Start training on " + self.device)
        train_metrics, validation_metrics = model.fit(
            training_dataloader,
            validation_dataloader,
            self.local_path(),
            self.device,
            nepochs=self.epochs + self.extend_training,
            resume_epochs=ran_epochs,
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
