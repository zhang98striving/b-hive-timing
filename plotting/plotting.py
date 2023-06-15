import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from BaseTask import MainBaseTask
from training.training import InferenceTask


class PlottingTask(MainBaseTask):
    def requires(self):
        return InferenceTask.req(self)

    def output(self):
        return self.local_target("plotting.txt")

    def run(self):
        input = np.load("input.npy", allow_pickle=True)
        output = np.load("output.npy", allow_pickle=True)
        plot_roc_curve(input, output)
        train_loss = np.load("train_metrics.npz", allow_pickle=True)["loss"]
        test_loss = np.load("test_metrics.npz", allow_pickle=True)["loss"]
        plot_losses(train_loss, test_loss)

        print("Done")
        self.output().dump("plotting", formatter="text")


def plot_roc_curve(input, output):
    b_jets = (input[:, -1, 0] == 0) | (input[:, -1, 0] == 1) | (input[:, -1, 0] == 2)
    prob_b = output[:, :3].sum(axis=1)
    print(prob_b.shape, prob_b[:10])

    c_veto = (input[:, -1, 0] != 3) & (input[:, 0, 0] > 30)  # id == 3 + jet_pt > 30
    light_veto = ((input[:, -1, 0] != 4) & (input[:, -1, 0] != 5)) & (
        input[:, 0, 0] > 30
    )  # id!=4 or !=5 + jet_pt>30

    fpr, tpr, _ = roc_curve(b_jets[c_veto], prob_b[c_veto])
    plt.plot(tpr, fpr, label="udsg")

    fpr, tpr, _ = roc_curve(b_jets[light_veto], prob_b[light_veto])
    plt.plot(tpr, fpr, label="c")

    plt.legend()
    plt.semilogy()
    plt.grid(alpha=0.4)
    plt.title("pt>30GeV, tt events")
    plt.xlabel("b jet efficiency")
    plt.ylabel("misid. probability")
    plt.savefig("roc.pdf")
    plt.close()


def plot_losses(train_loss, test_loss):
    plt.title("Losses")
    plt.plot(*np.array(list(enumerate(test_loss, 1))).T, label="Test")
    plt.plot(*np.array(list(enumerate(train_loss, 1))).T, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.pdf")
    plt.close()
