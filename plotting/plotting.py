import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import roc_curve

from BaseTask import MainBaseTask
from training.training import InferenceTask


class PlottingTask(MainBaseTask):
    def requires(self):
        return InferenceTask.req(self)

    def output(self):
        return self.local_target("loss.pdf")

    def run(self):
        input = np.load(self.output_directory + "/input.npy", allow_pickle=True)
        output = np.load(self.output_directory + "/output.npy", allow_pickle=True)
        plot_roc_curve(input, output, self.output_directory + "/")

        train_loss = np.load(self.output_directory + "/train_metrics.npz", allow_pickle=True)[
            "loss"
        ]
        test_loss = np.load(self.output_directory + "/test_metrics.npz", allow_pickle=True)["loss"]
        plot_losses(train_loss, test_loss, self.output_directory + "/")


def plot_roc_curve(input, output, output_dir):
    hep.style.use("CMS")
    hep.cms.text("")
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
