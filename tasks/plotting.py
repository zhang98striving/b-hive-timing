import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os

from rich.progress import track
from scipy.special import softmax
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from tasks.training import TrainingTask
from tasks.inference import InferenceTask

from utils.plotting.roc import prepare_roc, plot_losses
from utils.plotting.termplot import terminal_roc


class PlottingTask(TrainingDependency, DatasetDependency, BaseTask):
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

        predictions = np.load(self.input()["inference"]["prediction"].path, allow_pickle=True)
        kinematics = np.load(self.input()["inference"]["kinematics"].path, allow_pickle=True)
        truth = np.load(self.input()["inference"]["truth"].path, allow_pickle=True)
        pts = kinematics[..., 0]

        all_files = self.input()["dataset"]["file_list"].load()
        test_files = np.array([f for f in all_files if "test" in f])

        terminal_roc(predictions, truth)

        prepare_roc(
            test_files,
            self.local_path(),
            ["TT", "QCD"],
            truth,
            softmax(predictions, axis=-1),
            pts,
        )

        train_loss = np.load(self.input()["training"]["training_metrics"].path, allow_pickle=True)[
            "loss"
        ]
        validation_loss = np.load(
            self.input()["training"]["validation_metrics"].path, allow_pickle=True
        )["loss"]
        plot_losses(train_loss, validation_loss, self.local_path())
