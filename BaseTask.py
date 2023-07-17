from rich.console import Console
import luigi
import torch
import law
import os


c = Console()

# Creating a dictionary to store hyperparameters
config_dict = {}
config_dict["model"] = {}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    c.print("[black on yellow]Warning:", "No CUDA device available. Running on cpu...")
config_dict["device"] = device
# Defining the number of input parameters
config_dict["model"]["n_cpf"] = 25
config_dict["model"]["n_npf"] = 25
config_dict["model"]["n_vtx"] = 4


class MainBaseTask(law.Task):
    output_directory = luigi.Parameter(
        "/net/scratch/Matefarkas/phd/service_work/bhive_torch20/output"
    )
    fileformat = luigi.Parameter("numpy", description="Fileformat to use")
    loss_weighting = luigi.BoolParameter(
        True, description="Whether to weight the loss or use weighted sampling from the dataset"
    )

    chunk_size = 100000

    def local_path(self, *path):
        # DATA_PATH is defined in setup.sh
        parts = ("$DATA_PATH",) + path
        return os.path.join(*(str(p) for p in parts))

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)
