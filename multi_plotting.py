import os
import law
import torch
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
from sklearn.metrics import auc, roc_curve
from scipy.special import softmax
from utils.plotting.termplot import terminal_roc
from matplotlib.cm import get_cmap

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
    model_pu,
    xmin=0.0,
    energy="13.6 TeV",
    save_numpy=True,

):
    multi_num = len(truths)
    # The shape of discs, truths and vetos is 4*5*n. We have to swap the first and second dimensions.
    # 4 is the number of curves in one plot
    # 5 is the length of ['bvsall', 'bvsc', 'bvsl', 'cvsb', 'cvsl']
    swapped = []
    for j in range(len(discs[0])):  # 5
        temp = [] 
        for i in range(multi_num):  # 4
            temp.append(discs[i][j]) 
        swapped.append(temp)
    discs = swapped
    
    swapped = []
    for j in range(len(truths[0])):  # 5
        temp = [] 
        for i in range(multi_num):  # 4
            temp.append(truths[i][j]) 
        swapped.append(temp)
    truths = swapped
    
    swapped = []
    for j in range(len(vetos[0])):  # 5
        temp = []  
        for i in range(multi_num):  # 4
            temp.append(vetos[i][j]) 
        swapped.append(temp)
    vetos = swapped

    for disc, truth, veto, roc_label, xlabel, ylabel in zip(
        discs,
        truths,
        vetos,
        labels,
        xlabels,
        ylabels,
    ):
        fpr = []
        tpr = []
        area = []
        
        for i in range(multi_num):
            try:
                fpr_tmp, tpr_tmp, _ = roc_curve(truth[i][veto[i]], disc[i][veto[i]])
                fpr.append(fpr_tmp)
                tpr.append(tpr_tmp)
            except ValueError as e:
                print(e)
                print(
                    "Your ROC could not be plotted. Please check if this is not a debug set"
                )
                continue
            area.append( auc(fpr[i], tpr[i]) )
            if save_numpy:
                np.save(
                    os.path.join(output_directory, f"roc_default_{roc_label}_{i}.npy"),
                    np.array((fpr[i], tpr[i])),
                )
        for ext in [".png", ".pdf"]:
            plot_name = os.path.join(output_directory, f"roc_default_{roc_label}{ext}")
            
            plot_roc(
                fpr,
                tpr,
                area,
                [roc_label],
                name,
                pt_min=pt_min,
                pt_max=pt_max,
                x_label=xlabel,
                y_label=ylabel,
                output_path=plot_name,
                r_label=energy,
                xmin=xmin,
                model_pu = model_pu,
                )

def plot_roc(
    fpr,
    tpr,
    area,
    label_list,
    dataset_label=None,
    pt_min=None,
    pt_max=None,
    x_label="Tagging Efficiency",
    y_label="Mistagging rate",
    r_label=None,
    l_label="Preliminary",
    output_path="roc.pdf",
    xmin=None,
    model_pu=['DeepJet_noPU','DeepJet_noPU','DeepJet_noPU','DeepJet_noPU'],
):
    curve_count = len(model_pu)
    plt.figure()
    for label in label_list:
        for i in range(curve_count):
            area_tmp = area[i]    
            plt.plot(
                tpr[i],
                fpr[i],
                label=f"{label}_{model_pu[i]}" rf"(AUC${{\approx}}${np.round(area_tmp,3)})",
                color=color[i],
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
        title+=f"Test with {dataset_label} jets \n"
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



# path
local_path = '/home/home1/institut_3a/zhang/Documents/b-hive/phase2/multi_plots/mix_ttbar'
os.makedirs(local_path, exist_ok=True)

# config
config_name = ['offline_run3', 'offline_run3', 'part_run3', 'part_run3']
config = []
for i in range(len(config_name)):
    config.append( ConfigLoader.load_config(config_name[i]) )

# model
model_name = ['DeepJet', 'DeepJet', 'ParticleTransformer', 'ParticleTransformer']
model_name_short = ['DeepJet', 'DeepJet', 'ParT', 'ParT']
pileup = ['noPU', 'PU200', 'noPU', 'PU200']
model_pu = [p+'_'+q for p,q in zip(model_name_short,pileup)]
model = []
for i in range(len(model_name)):
    if issubclass(type(BTaggingModels(model_name[i])), torch.nn.Module):
        #model = BTaggingModels(model_name).to(self.device)
        model.append ( BTaggingModels(model_name[i]).to('cuda') )
    else:
        model.append ( BTaggingModels(model_name[i]) )

# color
color = ['green', 'blue', 'purple', 'red']

# predictions, truths, pts
Inference_path = [
    '/net/scratch_cms3a/zhang/b-hive/InferenceTask/offline_run3/Sep_mix_noPU/Sep_test_ttbar_noPU/Sep_training_mix_noPU/DeepJet/epochs_60/nominal/test_attack_nominal/',
    '/net/scratch_cms3a/zhang/b-hive/InferenceTask/offline_run3/Sep_mix_PU200/Sep_test_ttbar_PU200/Sep_training_mix_PU200/DeepJet/epochs_60/nominal/test_attack_nominal/',
    '/net/scratch_cms3a/zhang/b-hive/InferenceTask/part_run3/Sep_mix_noPU/Sep_test_ttbar_noPU/Sep_training_mix_noPU/ParticleTransformer/epochs_60/nominal/test_attack_nominal/',
    '/net/scratch_cms3a/zhang/b-hive/InferenceTask/part_run3/Sep_mix_PU200/Sep_test_ttbar_PU200/Sep_training_mix_PU200/ParticleTransformer/epochs_60/nominal/test_attack_nominal/',
    ]
path_count = len(Inference_path)

predictions_path = [s +'prediction.npy' for s in Inference_path]
kinematics_path = [s +'kinematics.npy' for s in Inference_path]
truth_path = [s +'truth.npy' for s in Inference_path]
process_path = [s +'process.npy' for s in Inference_path]

predictions = []
kinematics = []
truth = []
process = []
pts = []

for i in range(path_count):
    predictions.append( np.load( predictions_path[i] , allow_pickle=True) )
    kinematics.append(  np.load( kinematics_path[i] , allow_pickle=True) )
    truth.append( np.load( truth_path[i] , allow_pickle=True) )
    process.append( np.load( process_path[i] , allow_pickle=True) )
    pts.append( kinematics[i][..., 0] )
    
# b-hive/utils/plotting/termplot.py
for i in range(path_count):
    terminal_roc(predictions[i], truth[i])


proc = 'default'
proc_i = 0
proc_mask = []
pt_min = []
pt_max = []
mask = []
pt_mask = []
discs = []
truths = [] 
vetos = [] 
labels = [] 
xlabels = [] 
ylabels = []
for i in range(path_count):
    proc_mask.append( process[i] == proc_i )
    pt_min.append ( config[i].get(proc, {"pt_min": 0}).get("pt_min", 0) )
    pt_max.append( config[i].get(proc, {"pt_max": np.inf}).get("pt_max", np.inf) )
    pt_mask.append( np.logical_and(pts[i] > pt_min[i], pts[i] < pt_max[i]) )
    mask.append( np.logical_and(proc_mask[i], pt_mask[i]) )
    discs_tmp, truths_tmp, vetos_tmp, labels_tmp, xlabels_tmp, ylabels_tmp = model[i].calculate_roc_list(
        predictions[i][mask[i]], truth[i][mask[i]]
    )
    discs.append(discs_tmp)
    truths.append(truths_tmp)
    vetos.append(vetos_tmp)
    labels.append(labels_tmp)
    xlabels.append(xlabels_tmp)
    ylabels.append(ylabels_tmp)

plot_roc_list(
    discs=discs,
    truths=truths,
    vetos=vetos,
    labels=labels[0],
    xlabels=xlabels[0],
    ylabels=ylabels[0],
    output_directory=local_path,
    pt_min=pt_min,
    pt_max=pt_max,
    name='ttbar',
    model_pu = model_pu,
    xmin=0.4
)

plt.style.use(hep.cms.style.CMS)


