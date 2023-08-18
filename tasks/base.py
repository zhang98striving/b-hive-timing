from rich.console import Console
import luigi
import torch
import law
import os

c = Console()

# Creating a dictionary to store hyperparameters
config_dict = {}
config_dict["model"] = {}

# Defining the number of input parameters
config_dict["model"]["n_cpf"] = 25
config_dict["model"]["n_npf"] = 25
config_dict["model"]["n_vtx"] = 4


class MainBaseTask(law.Task):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        c.print("[black on yellow]Warning:", "No CUDA device available. Running on cpu...")

    def local_path(self, *path):
        # DATA_PATH is defined in setup.sh
        parts = (os.getenv("DATA_PATH"),) + path
        return os.path.join(*(str(p) for p in parts))

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)
