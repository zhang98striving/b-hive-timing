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


def plot_roc_list(
    discs,
    truths,
    vetos,
    labels,
    xlabels,
    ylabels,
    output_directory,
    pt_min,
    pt_max,
    name,
    energy="13.6 TeV",
    save_numpy=True,
):
    for disc, truth, veto, roc_label, xlabel, ylabel, color in zip(
        discs,
        truths,
        vetos,
        labels,
        xlabels,
        ylabels,
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
        plot_name = os.path.join(output_directory, f"roc_{name}_{roc_label}.pdf")
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


def plot_losses(train_loss, test_loss, output_dir=None, epochs=None):
    fig, ax = plt.subplots()
    ax.set_title("Losses")
    if train_loss is not None:
        ax.plot(np.linspace(0, epochs, len(train_loss)), train_loss, label="Validation", color="blue")
    if test_loss is not None:
        ax.plot(np.linspace(0,epochs, len(test_loss)), test_loss , label="Train", color="orange")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "loss.pdf"))
    fig.savefig(os.path.join(output_dir, "loss.png"))


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
    output_path="roc.pdf",
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
    plt.xlim(0.0, 1)
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
