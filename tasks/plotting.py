import os

import law
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from rich.progress import track
from scipy.special import softmax
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.inference import InferenceTask
from tasks.parameter_mixins import (
    DatasetDependency,
    TrainingDependency,
    TestDatasetDependency,
)
from tasks.training import TrainingTask
from utils.config.config_loader import ConfigLoader
from utils.plotting.roc import plot_roc_list, plot_losses
from utils.plotting.termplot import terminal_roc
from utils.models.models import BTaggingModels
import torch



class ROCCurveTask(
    TrainingDependency, TestDatasetDependency, DatasetDependency, BaseTask
):
    def requires(self):
        return {
            "training": TrainingTask.req(self),
            "inference": InferenceTask.req(self),
            "test_dataset": DatasetConstructorTask.req(
                self,
                dataset_version=self.test_dataset_version,
                filelist=self.test_filelist,
            ),
        }

    def output(self):
        return law.LocalDirectoryTarget(self.local_path())

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

        all_files = self.input()["test_dataset"]["file_list"].load()
        test_files = np.array([f for f in all_files if "test" in f])

        terminal_roc(predictions, truth)
        if issubclass(type(BTaggingModels(self.model_name)), torch.nn.Module):
            model = BTaggingModels(self.model_name).to(self.device)
        else:
            model = BTaggingModels(self.model_name)

        for proc_i, proc in enumerate(config["processes"]):
            print(f"Plotting ROC for {proc}")

            proc_mask = process == proc_i
            pt_min = config.get(proc, {"pt_min": 0}).get("pt_min", 0)
            pt_max = config.get(proc, {"pt_max": np.inf}).get("pt_max", np.inf)

            pt_mask = np.logical_and(pts > pt_min, pts < pt_max)
            mask = np.logical_and(proc_mask, pt_mask)

            discs, truths, vetos, labels, xlabels, ylabels = model.calculate_roc_list(
                predictions[mask], truth[mask]
            )

            plot_roc_list(
                discs=discs,
                truths=truths,
                vetos=vetos,
                labels=labels,
                xlabels=xlabels,
                ylabels=ylabels,
                output_directory=self.local_path(),
                pt_min=pt_min,
                pt_max=pt_max,
                name=proc,
            )
