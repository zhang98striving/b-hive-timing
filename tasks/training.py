import law
import luigi
import numpy as np
import os
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import AttackDependency, DatasetDependency, TrainingDependency
from utils.adversarial_attacks.pick_attack import pick_attack
from utils.config.config_loader import ConfigLoader
from utils.models.models import BTaggingModels
from utils.plotting.roc import plot_roc_list, plot_losses

law.contrib.load("numpy")


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


def load_resume_training(model, path, device, optimizer=None, epoch=None):
    try:
        model_path, ran_epochs = check_resume(path, load_epoch=epoch)
        _model = torch.load(
            model_path,
            map_location=torch.device(device),
        )
        model.load_state_dict(_model["model_state_dict"])
        if not (optimizer is None):
            optimizer.load_state_dict(_model["optimizer_state_dict"])
        print(f"Resuming on epoch {ran_epochs}:\n{model_path}")
        return model, optimizer, ran_epochs
    except FileNotFoundError:
        print("No training to resume found. Starting a new one.")
        return model, 0


class TrainingTask(AttackDependency, TrainingDependency, DatasetDependency, BaseTask):
    loss_weighting = luigi.BoolParameter(
        False,
        description="Whether to weight the loss or use weighted sampling from the dataset.",
    )

    resume_training = luigi.BoolParameter(
        False,
        description="Whether to resume the training if it already ran partially and failed. Set this to true if you want to resume.",
    )
    resume_epoch = luigi.IntParameter(
        False,
        description="Whether to resume the training from a specific epoch.",
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
            "model": (
                self.local_target(f"model_{self.epochs-1 + self.extend_training}.pt")
                if issubclass(type(BTaggingModels(self.model_name)), torch.nn.Module)
                else self.local_target(f"model_{self.epochs-1}.keras")
            ),
            "best_model": (
                self.local_target("best_model.pt")
                if issubclass(type(BTaggingModels(self.model_name)), torch.nn.Module)
                else self.local_target(f"best_model.keras")
            ),
        }

    def run(self):
        # Loading config
        config = ConfigLoader.load_config(self.config)
        os.makedirs(self.local_path(), exist_ok=True)
        print("Loading Dataset")
        files = self.input()["file_list"].load().split("\n")

        n_train = max((1, int( len(files) * self.train_val_split))) # has at least one training file
        training_files = files[:n_train]
        validation_files = files[n_train:]
        if len(validation_files) == 0:
             print("\nWARNING!")
             print("No validation files found. Please check your dataset. Most likely you only have one file!")
             print("Using the trainingfile for validation")
             print()
             validation_files = training_files
        if not( isinstance(training_files, list)):
            training_files = [training_files]
        if not( isinstance(validation_files, list)):
            validation_files = [validation_files]
        print(f"#Train files: {len(training_files)}")
        print(f"#Val files: {len(validation_files)}")

        histogram_training = np.load(
            self.input()["histogram"].path,
            allow_pickle=True,
        )

        # Model Defintion
        if issubclass(type(model := BTaggingModels(self.model_name)), torch.nn.Module):
            model = BTaggingModels(self.model_name).to(self.device)
            optimizer = model.optimizerClass(
                model.parameters(), lr=self.learning_rate, eps=1e-7
            )
        else:
            optimizer = model.optimizer

        # Picking attack
        print(
            rf"Will apply {self.attack} attack with epsilon={self.attack_magnitude} and {self.attack_iterations} iterations."
        )
        attack = pick_attack(
            self.attack,
            device=self.device,
            integer_positions=model.integers,
            default_values=model.defaults,
            epsilon=self.attack_magnitude,
            epsilon_factors=self.attack_individual_factors,
            iterations=self.attack_iterations,
            reduce=self.attack_reduce,
            restrict_impact=self.attack_restrict_impact,
        )

        print("Model construction")
        if self.resume_training or self.resume_epoch or self.extend_training:
            model, optimizer, ran_epochs = load_resume_training(
                model,
                self.local_path(),
                self.device,
                optimizer=optimizer,
                epoch=self.resume_epoch,
            )
            train_metrics_first = self.output()["training_metrics"].load()
            validation_metrics_first = self.output()["validation_metrics"].load()
        else:
            ran_epochs = 0
            train_metrics_first = {"loss": [], "acc": []}
            validation_metrics_first = {"loss": [], "acc": []}
        datasetClass = model.datasetClass
        # Define the training and validation datasets
        training_data = datasetClass(
            training_files,
            model=model,
            data_type="training",
            weighted_sampling=not (self.loss_weighting),

            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
            verbose=self.verbose,
            process_weights=[
                config.get("process-weights", {}).get(proc, 1.0)
                for proc in config.get("processes", [])
            ],
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
            process_weights=[
                config.get("process-weights", {}).get(proc, 1.0)
                for proc in config.get("processes", [])
            ],
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
        train_loss, val_loss, train_acc, val_acc = model.train_model(
            training_dataloader,
            validation_dataloader,
            self.local_path(),
            device=self.device,
            attack=attack,
            optimizer=optimizer,
            nepochs=self.epochs,
            resume_epochs=ran_epochs,
            attack_magnitude=self.attack_magnitude,
            attack_iterations=self.attack_iterations,
        )
        train_loss = np.concatenate((train_metrics_first["loss"], train_loss))
        train_acc = np.concatenate((train_metrics_first["acc"], train_acc))
        validation_loss = np.concatenate(
            (validation_metrics_first["loss"], val_loss)
        )
        validation_acc = np.concatenate(
            (validation_metrics_first["acc"], val_acc)
        )

        print("Training finished. Saving data...")

        np.savez(
            self.output()["training_metrics"].path,
            loss=train_loss,
            acc=train_acc,
            allow_pickle=True,
        )
        np.savez(
            self.output()["validation_metrics"].path,
            loss=validation_loss,
            acc=validation_acc,
            allow_pickle=True,
        )
        plot_losses(train_loss, val_loss, output_dir=self.local_path(), epochs=self.epochs+self.extend_training)
