from utils.torch.datasets import DeepJetDataset
from tasks.training import InferenceTask
from tasks.dataset import DatasetConstructorTask
from torch.utils.data import DataLoader
from tasks.base import BaseTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from utils.plotting.roc import prepare_roc, plot_losses
import matplotlib.pyplot as plt
from rich.progress import track
import mplhep as hep
import numpy as np
import os


class PlottingTask(TrainingDependency, DatasetDependency, BaseTask):
    def requires(self):
        return {
            "inference": InferenceTask.req(self),
            "dataset": DatasetConstructorTask.req(self),
        }

    def output(self):
        return self.local_target("loss.pdf")

    def run(self):
        os.makedirs(self.local_path(), exist_ok=True)
        files = np.array(
            open(self.input()["dataset"]["file_list"].path, "r").read().split("\n")[:-1]
        )
        test_mask = ~(np.char.find(files, "test") == -1)
        test_files = files[test_mask]
        histograms = np.load(self.input()["dataset"]["histogram_test"].path)
        test_data = DeepJetDataset(test_files, "test", histogram_training=histograms)
        test_dataloader = DataLoader(test_data, batch_size=1000)
        N_test_all = int(histograms.sum() / 2)
        input_data = np.empty((N_test_all))
        pts = np.empty((N_test_all))
        index = 0
        for x, _, y in track(test_dataloader, "Readin in predictions..."):
            pt = x[:, 0, 0]
            input_data[index : index + y.shape[0]] = y
            pts[index : index + y.shape[0]] = pt
            index += y.shape[0]
        output = np.load(self.input()["inference"]["output_numpy"].path, allow_pickle=True)
        sample_files = [
            os.path.join(self.input()["dataset"]["file_list"].parent.path, d)
            for d in os.listdir(self.input()["dataset"]["file_list"].parent.path)
            if d.endswith(".txt") and "test" in d
        ]
        samples_str_array = np.array([])
        for f in sample_files:
            samples_str_array = np.append(samples_str_array, open(f).read().split("\n")[:-2])

        prepare_roc(
            samples_str_array,
            self.local_path() + "/",
            ["TT", "QCD"],
            input_data,
            output,
            pts,
        )

        train_loss = np.load(self.local_path() + "/training_metrics.npz", allow_pickle=True)["loss"]
        validation_loss = np.load(self.local_path() + "/validation_metrics.npz", allow_pickle=True)[
            "loss"
        ]
        plot_losses(train_loss, validation_loss, self.local_path() + "/")
