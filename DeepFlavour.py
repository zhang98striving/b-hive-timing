import os

import law
import luigi
import numpy as np
import torch
import torch.nn.functional as F

# Import rich for pretty printing
from rich.console import Console
from torch.utils.data import DataLoader, random_split

from BaseTask import MainBaseTask

# from dataset.dataset import getDataset
from models.deepjet import DeepJet
from plotting.plotting import PlottingTask, plot_losses, plot_roc_curve
from training.training import InferenceTask, perform_training

c = Console()


class DeepJetRun(MainBaseTask):
    def requires(self):
        return PlottingTask.req(self)

    def output(self):
        return self.local_target("deepjetrun.txt")

    def run(self):
        c.print("Alles ready! Well done!")
