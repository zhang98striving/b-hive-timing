import os

import law
import matplotlib
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
    AttackDependency,
    DatasetDependency,
    TrainingDependency,
    TestAttackDependency,
    TestDatasetDependency,
)
from tasks.training import TrainingTask
from utils.config.config_loader import ConfigLoader
from utils.plotting.roc import plot_roc_list, plot_losses
from utils.plotting.termplot import terminal_roc
from utils.models.models import BTaggingModels
import torch

#from matplotlib.cm import get_cmap
from sklearn.metrics import auc, roc_curve
from scipy.special import softmax

from utils.plotting.termplot import terminal_roc
from matplotlib.cm import get_cmap



def plot_roc_list(
    discs1,
    discs2,
    truths1,
    truths2,
    vetos1,
    vetos2,
    labels,
    xlabels,
    ylabels,
    output_directory,
    pt_min1,
    pt_min2,
    pt_max1,
    pt_max2,
    name,
    model_name1 = 'DeepJet',
    model_name2 = 'ParticleTransformer',
    xmin=0.0,
    energy="13.6 TeV",
    save_numpy=True,

):
    for disc1, truth1, veto1, roc_label, disc2, truth2, veto2, xlabel, ylabel in zip(
        discs1,
        truths1,
        vetos1,
        labels,
        discs2,
        truths2,
        vetos2,
        xlabels,
        ylabels,
    ):
        try:
            fpr1, tpr1, _ = roc_curve(truth1[veto1], disc1[veto1])
            fpr2, tpr2, _ = roc_curve(truth2[veto2], disc2[veto2])
        except ValueError as e:
            print(e)
            print(
                "Your ROC could not be plotted. Please check if this is not a debug set"
            )
            continue
        area1 = auc(fpr1, tpr1)
        area2 = auc(fpr2, tpr2)
        if save_numpy:
            np.save(
                os.path.join(output_directory, f"roc_{name}_{roc_label}_1.npy"),
                np.array((fpr1, tpr1)),
            )
            np.save(
                os.path.join(output_directory, f"roc_{name}_{roc_label}_2.npy"),
                np.array((fpr2, tpr2)),
            )
        for ext in [".png", ".pdf"]:
            plot_name = os.path.join(output_directory, f"roc_{name}_{roc_label}{ext}")
            
            plot_roc(
                [(fpr1, tpr1, area1)],
                [(fpr2, tpr2, area2)],
                [roc_label],
                name,
                pt_min1=pt_min1,
                pt_min2=pt_min2,
                pt_max1=pt_max1,
                pt_max2=pt_max2,
                x_label=xlabel,
                y_label=ylabel,
                output_path=plot_name,
                r_label=energy,
                xmin=xmin,
                model_name1 = 'DeepJet',
                model_name2 = 'ParticleTransformer',
            )


# adapted from https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.py and https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.ipynb
'''
def calculate_roc(truth, discriminator, veto, output_directory, dataset_key, name):
    fpr, tpr, _ = roc_curve(truth[veto], discriminator[veto])

    index = np.unique(fpr, return_index=True)[1]
    fpr = np.asarray([fpr[i] for i in sorted(index)])
    tpr = np.asarray([tpr[i] for i in sorted(index)])
    area = auc(fpr, tpr)
    return fpr, tpr, area
'''

'''
def plot_losses(train_loss, test_loss, output_dir=None, epochs=None):
    fig, ax = plt.subplots()
    ax.set_title("Losses")
    if train_loss is not None:
        ax.plot(np.linspace(0, epochs, len(train_loss)), train_loss, label="Train", color="blue")
    if test_loss is not None:
        ax.plot(np.linspace(0,epochs, len(test_loss)), test_loss , label="Validation", color="orange")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "loss.pdf"))
    fig.savefig(os.path.join(output_dir, "loss.png"))
'''

def plot_roc(
    roc_list1,
    roc_list2,
    label_list,
    dataset_label=None,
    pt_min1=None,
    pt_min2=None,
    pt_max1=None,
    pt_max2=None,
    x_label="Tagging Efficiency",
    y_label="Mistagging rate",
    r_label=None,
    l_label="Preliminary",
    output_path="roc.pdf",
    xmin=None,
    model_name1 = 'DeepJet',
    model_name2 = 'ParticleTransformer',
):
    if not (isinstance(roc_list1, list)):
        roc_list1 = [roc_list1]
    if not (isinstance(roc_list2, list)):
        roc_list2 = [roc_list2]
    if not (isinstance(label_list, list)):
        label_list = [label_list]
    
    '''
    if colors1 is None:
        colors1 = color_set_list[: len(roc_list1)]
    if not (isinstance(colors1, list)):
        colors1 = [colors1]
    if len(colors1) < len(roc_list1):
        colors1 *= len(roc_list1)

    if colors2 is None:
        colors2 = color_set_list[: len(roc_list2)]
    if not (isinstance(colors2, list)):
        colors2 = [colors2]
    if len(colors2) < len(roc_list2):
        colors2 *= len(roc_list2)
    '''
    
    #pt_text1 = rf"${pt_min1} \leq p_T \leq {pt_max1}\,GeV$"
    #pt_text2 = rf"${pt_min2} \leq p_T \leq {pt_max2}\,GeV$"
    #eta_text = rf"$|\eta| \leq 2.5$"

    plt.figure()
    for roc1, roc2, label in zip(roc_list1, roc_list2, label_list):
        try:
            fpr1, tpr1, area1 = roc1
        except ValueError as e:
            from sklearn.metrics import auc # for some reason it could not find auc without another import, but I do not see why.
            fpr1, tpr1 = roc1
            index = np.unique(fpr1, return_index=True)[1]
            fpr1 = np.asarray([fpr1[i] for i in sorted(index)])
            tpr1 = np.asarray([tpr1[i] for i in sorted(index)])
            area1 = auc(fpr1, tpr1)
        try:
            fpr2, tpr2, area2 = roc2
        except ValueError as e:
            from sklearn.metrics import auc 
            fpr2, tpr2 = roc2
            index = np.unique(fpr2, return_index=True)[1]
            fpr2 = np.asarray([fpr2[i] for i in sorted(index)])
            tpr2 = np.asarray([tpr2[i] for i in sorted(index)])
            area2 = auc(fpr2, tpr2)
        plt.plot(
            tpr1,
            fpr1,
            label=f"{label}_{model_name1}" rf"(AUC${{\approx}}${np.round(area1, 3)})",
            color=color1,
        )
        plt.plot(
            tpr2,
            fpr2,
            label=f"{label}_{model_name2}" + rf"(AUC${{\approx}}${np.round(area2, 3)})",
            color=color2,
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yscale("log")
    plt.xlim(xmin, 1)
    plt.ylim(2 * 1e-4, 1)
    plt.grid(which="minor", alpha=0.85)
    plt.grid(which="major", alpha=0.95, color="black")
    title = ""
    
    if dataset_label:
        title+=f"{dataset_label} jets \n"
    #if pt_min and pt_max:
        #title+=f"{pt_text}, {eta_text}"
    plt.legend(
        title=title, 
        loc="best",
        alignment="left",
    )
    hep.cms.label(l_label, rlabel=r_label, com=13)

    print("saving to:\t", output_path)
    plt.savefig(output_path)
    plt.close()




local_path = '/home/home1/institut_3a/zhang/Documents/b-hive/phase2/multi_plots'
os.makedirs(local_path, exist_ok=True)


config_name1 = 'offline_run3'
config_name2 = 'part_run3'
config1 = ConfigLoader.load_config(config_name1)
config2 = ConfigLoader.load_config(config_name2)
'''
predictions_path1 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/offline_run3/T1234/T1234_test/T1234_training01/DeepJet/epochs_10/nominal/test_attack_nominal/prediction.npy'
kinematics_path1 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/offline_run3/T1234/T1234_test/T1234_training01/DeepJet/epochs_10/nominal/test_attack_nominal/kinematics.npy'
truth_path1 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/offline_run3/T1234/T1234_test/T1234_training01/DeepJet/epochs_10/nominal/test_attack_nominal/truth.npy'
process_path1 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/offline_run3/T1234/T1234_test/T1234_training01/DeepJet/epochs_10/nominal/test_attack_nominal/process.npy'

predictions_path2 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/part_run3/T2345/T2345_test/T2345_training01/ParticleTransformer/epochs_10/nominal/test_attack_nominal/prediction.npy'
kinematics_path2 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/part_run3/T2345/T2345_test/T2345_training01/ParticleTransformer/epochs_10/nominal/test_attack_nominal/kinematics.npy'
truth_path2 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/part_run3/T2345/T2345_test/T2345_training01/ParticleTransformer/epochs_10/nominal/test_attack_nominal/truth.npy'
process_path2 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/part_run3/T2345/T2345_test/T2345_training01/ParticleTransformer/epochs_10/nominal/test_attack_nominal/process.npy'
'''

Inference_path1 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/offline_run3/Sep_mix_PU200/Sep_test_ttbar_PU200/Sep_training_mix_PU200/DeepJet/epochs_60/nominal/test_attack_nominal/'
predictions_path1 = Inference_path1 +'prediction.npy'
kinematics_path1 = Inference_path1 +'kinematics.npy'
truth_path1 = Inference_path1 +'truth.npy'
process_path1 = Inference_path1 +'process.npy'

Inference_path2 = '/net/scratch_cms3a/zhang/b-hive/InferenceTask/part_run3/Sep_mix_PU200/Sep_test_ttbar_PU200/Sep_training_mix_PU200/ParticleTransformer/epochs_60/nominal/test_attack_nominal/'
predictions_path2 = Inference_path2 +'prediction.npy'
kinematics_path2 = Inference_path2 +'kinematics.npy'
truth_path2 = Inference_path2 +'truth.npy'
process_path2 = Inference_path2 +'process.npy'

predictions1 = np.load( predictions_path1 , allow_pickle=True)
kinematics1 = np.load( kinematics_path1 , allow_pickle=True)
truth1 = np.load( truth_path1 , allow_pickle=True)
process1 = np.load( process_path1 , allow_pickle=True)
pts1 = kinematics1[..., 0]

predictions2 = np.load( predictions_path2 , allow_pickle=True)
kinematics2 = np.load( kinematics_path2 , allow_pickle=True)
truth2 = np.load( truth_path2 , allow_pickle=True)
process2 = np.load( process_path2 , allow_pickle=True)
pts2 = kinematics2[..., 0]

#all_files = self.input()["test_dataset"]["file_list"].load()
#test_files = np.array([f for f in all_files if "test" in f])


#color_set_name1 = "Dark2"
#cmap1 = matplotlib.colormaps[color_set_name1]  # type: matplotlib.colors.ListedColormap
#color_set_list1 = cmap1.colors  # type: list
#color1 = color_set_list1[0]
color1 = 'green'

#color_set_name2 = "Accent"
#cmap2 = matplotlib.colormaps[color_set_name2]  # type: matplotlib.colors.ListedColormap
#color_set_list2 = cmap2.colors  # type: list
#color2 = color_set_list2[0]
color2 = 'blue'

model_name1 = 'DeepJet'
model_name2 = 'ParticleTransformer'

terminal_roc(predictions1, truth1)
terminal_roc(predictions2, truth2)

if issubclass(type(BTaggingModels(model_name1)), torch.nn.Module):
    #model = BTaggingModels(model_name).to(self.device)
    model1 = BTaggingModels(model_name1).to('cuda')
else:
    model1 = BTaggingModels(model_name1)
    
if issubclass(type(BTaggingModels(model_name2)), torch.nn.Module):
    #model = BTaggingModels(model_name).to(self.device)
    model2 = BTaggingModels(model_name2).to('cuda')
else:
    model2 = BTaggingModels(model_name2)

#for proc_i, proc in enumerate(config["processes"]):
#    print(f"Plotting ROC for {proc}")

proc = 'default'
proc_i = 0


proc_mask1 = process1 == proc_i
proc_mask2 = process2 == proc_i
pt_min1 = config1.get(proc, {"pt_min": 0}).get("pt_min", 0)
pt_max1 = config1.get(proc, {"pt_max": np.inf}).get("pt_max", np.inf)

pt_min2 = config2.get(proc, {"pt_min": 0}).get("pt_min", 0)
pt_max2 = config2.get(proc, {"pt_max": np.inf}).get("pt_max", np.inf)

pt_mask1 = np.logical_and(pts1 > pt_min1, pts1 < pt_max1)
pt_mask2 = np.logical_and(pts2 > pt_min2, pts2 < pt_max2)

mask1 = np.logical_and(proc_mask1, pt_mask1)
mask2 = np.logical_and(proc_mask2, pt_mask2)

discs1, truths1, vetos1, labels1, xlabels1, ylabels1 = model1.calculate_roc_list(
    predictions1[mask1], truth1[mask1]
)
discs2, truths2, vetos2, labels2, xlabels2, ylabels2 = model2.calculate_roc_list(
    predictions2[mask2], truth2[mask2]
)

plot_roc_list(
    discs1=discs1,
    discs2=discs2,
    truths1=truths1,
    truths2=truths2,
    vetos1=vetos1,
    vetos2=vetos2,
    labels=labels1,
    xlabels=xlabels1,
    ylabels=ylabels1,
    output_directory=local_path,
    pt_min1=pt_min1,
    pt_min2=pt_min2,
    pt_max1=pt_max1,
    pt_max2=pt_max2,
    name=proc,
    model_name1 = 'DeepJet',
    model_name2 = 'ParticleTransformer',
    xmin=0.4
)



plt.style.use(hep.cms.style.CMS)


