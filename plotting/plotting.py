import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from rich.progress import track
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from BaseTask import MainBaseTask
from training.training import DeepJetDataset, InferenceTask


class PlottingTask(MainBaseTask):
    def requires(self):
        return InferenceTask.req(self)

    def output(self):
        return self.local_target("loss.pdf")

    def run(self):
        # input = np.load(self.output_directory + "/input.npy", allow_pickle=True)
        files = np.array(
            open(f"{self.output_directory}/processed_files.txt", "r").read().split("\n")[:-1]
        )
        test_mask = ~(np.char.find(files, "test") == -1)
        test_files = files[test_mask]
        histograms = np.load(f"{self.output_directory}/test_data_histograms.npy")
        test_data = DeepJetDataset(test_files, "test")
        test_dataloader = DataLoader(test_data, batch_size=1000)
        N_test_all = int(histograms.sum() / 2)
        input_data = np.empty((N_test_all))
        pts = np.empty((N_test_all))
        index = 0
        for (x, _, y) in track(test_dataloader, "Readin in predictions..."):
            pt = x[:, 0, 0]
            with torch.no_grad():
                input_data[index : index + y.shape[0]] = y
                pts[index : index + y.shape[0]] = pt
                index += y.shape[0]
        print(N_test_all, index)
        output = np.load(self.output_directory + "/output.npy", allow_pickle=True)
        sample_files = [
            "/net/scratch/Matefarkas/phd/service_work/weight_eval_before_right_processing_run3/output/"
            + d
            for d in os.listdir(
                "/net/scratch/Matefarkas/phd/service_work/weight_eval_before_right_processing_run3/output/"
            )
            if ".txt" in d and d != "processed_files.txt" and "test" in d
        ]
        samples_str_array = np.array([])
        for f in sample_files:
            samples_str_array = np.append(samples_str_array, open(f).read().split("\n")[:-2])
        tt_samples_mask = ~(np.char.find(samples_str_array, "TT") == -1)
        input_data = input_data[tt_samples_mask]
        pts = pts[tt_samples_mask]
        output = output[tt_samples_mask]

        plot_roc_curve(input_data, output, pts, self.output_directory + "/")

        train_loss = np.load(self.output_directory + "/train_metrics.npz", allow_pickle=True)[
            "loss"
        ]
        test_loss = np.load(self.output_directory + "/test_metrics.npz", allow_pickle=True)["loss"]
        plot_losses(train_loss, test_loss, self.output_directory + "/")


def plot_roc_curve(input, output, pts, output_dir):
    hep.style.use("CMS")
    hep.cms.text("")
    b_jets = (input == 0) | (input == 1) | (input == 2)
    prob_b = output[:, :3].sum(axis=1)
    print(prob_b.shape, prob_b[:10])

    c_veto = (input != 3) & (pts > 30)  # id == 3 + jet_pt > 30
    light_veto = ((input != 4) & (input != 5)) & (pts > 30)  # id!=4 or !=5 + jet_pt>30

    fpr, tpr, _ = roc_curve(b_jets[c_veto], prob_b[c_veto])
    plt.plot(tpr, fpr, label="udsg")

    fpr, tpr, _ = roc_curve(b_jets[light_veto], prob_b[light_veto])
    plt.plot(tpr, fpr, label="c")

    plt.legend()
    plt.semilogy()
    plt.grid(alpha=0.4)
    plt.title(r"pt>30GeV, t$\bar{t}$ events")
    plt.xlabel("b jet efficiency")
    plt.ylabel("misid. probability")
    plt.savefig(output_dir + "roc.pdf")
    plt.close()


def plot_losses(train_loss, test_loss, output_dir):
    plt.title("Losses")
    plt.plot(*np.array(list(enumerate(test_loss, 1))).T, label="Test")
    plt.plot(*np.array(list(enumerate(train_loss, 1))).T, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_dir + "loss.pdf")
    plt.close()
