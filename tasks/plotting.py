import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from rich.progress import track
from scipy.special import softmax
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.inference import InferenceTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from tasks.training import TrainingTask
from utils.config.config_loader import ConfigLoader
from utils.plotting.roc import plot_all_rocs, plot_losses
from utils.plotting.termplot import terminal_roc


class ROCCurveTask(TrainingDependency, DatasetDependency, BaseTask):
    def requires(self):
        return {
            "training": TrainingTask.req(self),
            "inference": InferenceTask.req(self),
            "dataset": DatasetConstructorTask.req(self),
        }

    def output(self):
        return self.local_target("loss.pdf")

    def run(self):
        os.makedirs(self.local_path(), exist_ok=True)
        config = ConfigLoader.load_config(self.config)

        predictions = np.load(
            self.input()["inference"]["prediction"].path, allow_pickle=True
        )
        kinematics = np.load(
            self.input()["inference"]["kinematics"].path, allow_pickle=True
        )
        truth = np.load(self.input()["inference"]["truth"].path, allow_pickle=True)
        process = np.load(self.input()["inference"]["process"].path, allow_pickle=True)
        pts = kinematics[..., 0]

        all_files = self.input()["dataset"]["file_list"].load()
        test_files = np.array([f for f in all_files if "test" in f])

        terminal_roc(predictions, truth)

        for proc_i, proc in enumerate(config["processes"]):
            print(f"Plotting ROC for {proc}")
            proc_mask = process == proc_i
            pt_mask = np.logical_and(
                pts > config[proc].get("pt_min", 0),
                pts < config[proc].get("pt_max", 9999999),
            )
            mask = np.logical_and(proc_mask, pt_mask)
            plot_all_rocs(
                predictions[mask],
                truth[mask],
                self.local_path(),
                pt_min=config[proc].get("pt_min", None),
                pt_max=config[proc].get("pt_max", None),
                name=proc,
            )
        train_loss = np.load(
            self.input()["training"]["training_metrics"].path, allow_pickle=True
        )["loss"]
        validation_loss = np.load(
            self.input()["training"]["validation_metrics"].path, allow_pickle=True
        )["loss"]
        plot_losses(train_loss, validation_loss, self.local_path())
