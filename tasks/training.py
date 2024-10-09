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
from utils.plotting.roc import plot_roc_list, plot_losses, plot_accuracy
from utils.optimizing.SchedulerLoader import SchedulerLoader

law.contrib.load("numpy")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
            return models[max_epoch], max_epoch + 1, models
        else:
            return models[load_epoch - 1], load_epoch, models


def load_resume_training(model, path, device, output, optimizer=None, scheduler=None, epoch=None):
    try:
        model_path, ran_epochs, models = check_resume(path, load_epoch=epoch)
        _model = torch.load(
            model_path,
            map_location=torch.device(device),
        )
        model.load_state_dict(_model["model_state_dict"])
        if not (optimizer is None):
            optimizer.load_state_dict(_model["optimizer_state_dict"])
        if not (scheduler is None):
            scheduler.load_state_dict(_model["scheduler_state_dict"])
        print(f"Resuming on epoch {ran_epochs}:\n{model_path}")
    except FileNotFoundError:
        print("No training to resume found. Starting a new one")
        ran_epochs = 0
        best_loss_val = np.inf
    
    train_metrics = {"loss": [], "acc": []}
    validation_metrics = {"loss": [], "acc": []}
    
    if ran_epochs != 0:
        try:
            train_metrics = output["training_metrics"].load()
            train_metrics = {key: value.tolist() for key, value in train_metrics.items()}
            validation_metrics = output["validation_metrics"].load()
            validation_metrics = {key: value.tolist() for key, value in validation_metrics.items()}
        except FileNotFoundError:
            print("Training and validation metrics were not saved correctly during previous run. Recovering metrics from .pt files.")
            for epoch in range(ran_epochs):
                checkpoint = torch.load(models[epoch])
                train_metrics['loss'].append(checkpoint["loss_train"])
                train_metrics['acc'].append(checkpoint["acc_train"])
                validation_metrics['loss'].append(checkpoint["loss_val"])
                validation_metrics['acc'].append(checkpoint["acc_val"])
        best_loss_val = min(validation_metrics["loss"])        
                
    return model, optimizer, scheduler, ran_epochs, train_metrics, validation_metrics, best_loss_val


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
            model = model.to(self.device)
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

        print("Dataset construction")
        
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

        # The learning rate scheduler
        scheduler, batch_lr =  SchedulerLoader(
            self.lr_scheduler, 
            optimizer, 
            self.epochs, 
            dataloader = training_dataloader
        )

        print("Model construction")
        print(self.model_name)
        
        if self.resume_training or self.resume_epoch or self.extend_training:
            model, optimizer, scheduler, ran_epochs, train_metrics, validation_metrics, best_loss_val = load_resume_training(
                model,
                self.local_path(),
                self.device,
                self.output(),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=self.resume_epoch,
            )
        else:
            ran_epochs = 0
            train_metrics = {"loss": [], "acc": []}
            validation_metrics = {"loss": [], "acc": []}
            best_loss_val = np.inf

        # TO BE IMPLEMENTED
        #print(summary(model, input_size=model.input_dim_torchinfo, device = self.device))
        
        print(f'Total number of parameters: {count_parameters(model):,}')
        
        # Training
        print("Start training on " + self.device)
        train_metrics, validation_metrics = model.train_model(
            training_dataloader,
            validation_dataloader,
            self.local_path(),
            device=self.device,
            attack=attack,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_lr=batch_lr,
            best_loss_val=best_loss_val,
            nepochs=self.epochs,
            resume_epochs=ran_epochs,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )
        
        plot_losses(train_metrics['loss'], validation_metrics['loss'], output_dir=self.local_path())
        plot_accuracy(train_metrics['acc'], validation_metrics['acc'], output_dir=self.local_path())
