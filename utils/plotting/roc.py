import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.metrics import auc, roc_curve
from scipy.special import softmax

from utils.plotting.termplot import terminal_roc
from matplotlib.cm import get_cmap

color_set_name = "Dark2"
cmap = get_cmap(color_set_name)  # type: matplotlib.colors.ListedColormap
color_set_list = cmap.colors  # type: list

plt.style.use(hep.cms.style.CMS)


def plot_all_rocs(
    predictions,
    truth,
    output_directory,
    pt_min,
    pt_max,
    name,
    energy="13.6 TeV",
    save_numpy=True,
):
    if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
        predictions = softmax(predictions, axis=-1)

    b_jets = (truth == 0) | (truth == 1) | (truth == 2)
    c_jets = truth == 3
    l_jets = (truth == 4) | (truth == 5)
    summed_jets = b_jets + c_jets + l_jets

    b_pred = predictions[:, :3].sum(axis=1)
    c_pred = predictions[:, 3]
    l_pred = predictions[:, -2:].sum(axis=1)

    bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
    bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
    cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
    cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)
    bvsall = np.where(
        (b_pred + l_pred + c_pred) > 0, (b_pred) / (b_pred + l_pred + c_pred), -1
    )

    b_veto = (truth != 0) & (truth != 1) & (truth != 2) & (summed_jets != 0)
    c_veto = (truth != 3) & (summed_jets != 0)
    l_veto = (truth != 4) & (truth != 5) & (summed_jets != 0)
    no_veto = np.ones(b_veto.shape, dtype=np.bool)

    for roc_label, disc, veto, truth, xlabel, ylabel, color in zip(
        ["bvsl", "bvsc", "cvsb", "cvsl", "bvsall"],
        [bvsl, bvsc, cvsb, cvsl, bvsall],
        [c_veto, l_veto, b_veto, b_veto, no_veto],
        [b_jets, b_jets, c_jets, c_jets, b_jets],
        [
            "b-identification",
            "b-identification",
            "c-identification",
            "c-identification",
            "b-identification",
        ],
        ["light mis-id.", "c mis-id", "b mis-id.", "light mis-id.", "mis-id."],
        color_set_list[0:5],
    ):
        try:
            fpr, tpr, _ = roc_curve(truth[veto], disc[veto])
        except ValueError as e:
            print(e)
            print(
                "Your ROC could not be plotted. Please check if this is not a debug set"
            )
            continue
        area = auc(fpr, tpr)
        plot_name = os.path.join(output_directory, f"roc_{name}_{roc_label}.jpg")
        if save_numpy:
            np.save(
                os.path.join(output_directory, f"roc_{name}_{roc_label}.npy"),
                np.array((fpr, tpr)),
            )
        plot_roc(
            [(fpr, tpr, area)],
            [roc_label],
            name,
            pt_min=pt_min,
            pt_max=pt_max,
            x_label=xlabel,
            y_label=ylabel,
            output_path=plot_name,
            colors=color,
            r_label=energy,
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
    if not (isinstance(roc_list, list)):
        roc_list = [roc_list]
    if not (isinstance(label_list, list)):
        label_list = [label_list]
    if colors is None:
        colors = color_set_list[: len(roc_list)]
    if not (isinstance(colors, list)):
        colors = [colors]
    if len(colors) < len(roc_list):
        colors *= len(roc_list)

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
