import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.metrics import auc, roc_curve

from utils.plotting.termplot import terminal_roc
from matplotlib.cm import get_cmap

color_set_name = "Dark2"
cmap = get_cmap(color_set_name)  # type: matplotlib.colors.ListedColormap
color_set_list = cmap.colors  # type: list

color_set_name = "Dark2"
cmap = get_cmap(color_set_name)  # type: matplotlib.colors.ListedColormap
color_set_list = cmap.colors  # type: list

plt.style.use(hep.cms.style.CMS)


# adapted from https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.py and https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.ipynb
def prepare_roc(sample_names, output_directory, dataset_keys, truth, output_data, jet_pt):
    for key in dataset_keys:
        sample_mask = ~(np.char.find(sample_names, key) == -1)

        truth_ = truth[sample_mask]
        output_data_ = output_data[sample_mask]
        jet_pt_ = jet_pt[sample_mask]

        if key == "TT":
            pt_min = 30
            pt_max = 1000
        elif key == "QCD":
            pt_min = 30
            pt_max = 1000
        else:
            raise NotImplementedError("Wrong dataset typ.")

        jet_mask = (jet_pt_ > pt_min) & (jet_pt_ < pt_max)

        output_data_ = output_data_[jet_mask]
        truth_ = truth_[jet_mask]

        b_jets = (truth_ == 0) | (truth_ == 1) | (truth_ == 2)
        c_jets = truth_ == 3
        l_jets = (truth_ == 4) | (truth_ == 5)
        summed_jets = b_jets + c_jets + l_jets

        b_pred = output_data_[:, :3].sum(axis=1)
        c_pred = output_data_[:, 3]
        l_pred = output_data_[:, -2:].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)

        b_veto = (truth_ != 0) & (truth_ != 1) & (truth_ != 2) & (summed_jets != 0)
        c_veto = (truth_ != 3) & (summed_jets != 0)
        l_veto = (truth_ != 4) & (truth_ != 5) & (summed_jets != 0)

        if len(b_jets) == 0:
            print("Skipping...")
            continue

        roc_list = []
        label_list = ["BvsL", "CvsB", "CvsL", "BvsC"]
        roc_list.append(calculate_roc(b_jets, bvsl, c_veto, output_directory, key, label_list[0]))
        roc_list.append(calculate_roc(c_jets, cvsb, l_veto, output_directory, key, label_list[1]))
        roc_list.append(calculate_roc(c_jets, cvsl, b_veto, output_directory, key, label_list[2]))
        roc_list.append(calculate_roc(b_jets, bvsc, l_veto, output_directory, key, label_list[3]))

        plot_roc(
            roc_list,
            label_list,
            key,
            pt_min=pt_min,
            pt_max=pt_max,
            output_path=os.path.join(output_directory, "roc.png"),
        )


# adapted from https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.py and https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.ipynb
def calculate_roc(truth, discriminator, veto, output_directory, dataset_key, name):
    fpr, tpr, _ = roc_curve(truth[veto], discriminator[veto])

    index = np.unique(fpr, return_index=True)[1]
    fpr = np.asarray([fpr[i] for i in sorted(index)])
    tpr = np.asarray([tpr[i] for i in sorted(index)])
    area = auc(fpr, tpr)
    return fpr, tpr, area


def plot_losses(train_loss, test_loss, output_dir):
    plt.title("Losses")
    plt.plot(*np.array(list(enumerate(test_loss, 1))).T, label="Test")
    plt.plot(*np.array(list(enumerate(train_loss, 1))).T, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.pdf"))
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()


def plot_roc(
    roc_list,
    label_list,
    dataset_label=None,
    pt_min=None,
    pt_max=None,
    x_label="Tagging Efficiency",
    y_label="Mistagging rate",
    r_label=None,
    l_label="Preliminary",
    output_path="roc.png",
    colors=None,
):
    if colors is None:
        colors = color_set_list[: len(roc_list)]

    pt_text = rf"${pt_min} \leq p_T \leq {pt_max}\,GeV$"
    eta_text = rf"$|\eta| \leq 2.5$"

    plt.figure()
    for roc, label, color in zip(roc_list, label_list, colors):
        fpr, tpr, auc = roc
        plt.plot(
            tpr,
            fpr,
            label=f"{label}" + rf"(AUC${{\approx}}${np.round(auc, 3)})",
            color=color,
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yscale("log")
    plt.xlim(0.4, 1)
    plt.ylim(2 * 1e-4, 1)
    plt.grid(which="minor", alpha=0.85)
    plt.grid(which="major", alpha=0.95, color="black")
    plt.legend(
        title=f"{dataset_label} jets \n {pt_text}, {eta_text}",
        loc="best",
        alignment="left",
    )
    hep.cms.label(l_label, rlabel=r_label, com=13)

    print("saving to:\t", output_path)
    plt.savefig(output_path)
    plt.close()
