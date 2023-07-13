import os

import law
import luigi
import numpy as np
import torch
import torch.nn.functional as F

# Import rich for pretty printing
from rich.console import Console

from BaseTask import MainBaseTask
from plotting.plotting import PlottingTask

c = Console()


class DeepJetRun(MainBaseTask):
    def requires(self):
        return PlottingTask.req(self)

    def output(self):
        return self.local_target("deepjetrun.txt")

    def run(self):
        c.print("Alles ready! Well done!")
