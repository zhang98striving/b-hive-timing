from plotting.plotting import PlottingTask
from BaseTask import MainBaseTask
from rich.console import Console
import torch.nn.functional as F
import numpy as np
import luigi
import torch
import law
import os


c = Console()


class DeepJetRun(MainBaseTask):
    def requires(self):
        return PlottingTask.req(self)

    def output(self):
        return self.local_target("deepjetrun.txt")

    def run(self):
        c.print("Alles ready! Well done!")
