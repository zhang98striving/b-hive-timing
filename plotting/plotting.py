from training.training import DeepJetDataset, InferenceTask
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from BaseTask import MainBaseTask
import matplotlib.pyplot as plt
from rich.progress import track
import mplhep as hep
import numpy as np
import os

plt.style.use(hep.cms.style.ROOT)


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
        for x, _, y in track(test_dataloader, "Readin in predictions..."):
            pt = x[:, 0, 0]
            input_data[index : index + y.shape[0]] = y
            pts[index : index + y.shape[0]] = pt
            index += y.shape[0]
        output = np.load(self.output_directory + "/output.npy", allow_pickle=True)
        sample_files = [
            self.output_directory + "/" + d
            for d in os.listdir(self.output_directory)
            if ".txt" in d and d != "processed_files.txt" and "test" in d
        ]
        samples_str_array = np.array([])
        for f in sample_files:
            samples_str_array = np.append(samples_str_array, open(f).read().split("\n")[:-2])
        prepare_roc(samples_str_array, self.output_directory + "/", ["TT", "QCD"], input_data, output, pts)

        train_loss = np.load(self.output_directory + "/training_metrics.npz", allow_pickle=True)[
            "loss"
        ]
        validation_loss = np.load(self.output_directory + "/validation_metrics.npz", allow_pickle=True)["loss"]
        plot_losses(train_loss, validation_loss, self.output_directory + "/")

# adapted from https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.py and https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.ipynb
def prepare_roc(input_directory, output_directory, dataset_keys, truth, output_data, jet_pt):
    for key in dataset_keys:
        sample_mask = ~(np.char.find(input_directory, key) == -1)
        truth_       = truth[sample_mask]
        output_data_ = output_data[sample_mask]
        jet_pt_      = jet_pt[sample_mask]

        if key=="TT":
            pt_min = 30
            pt_max = 1000
        elif key=="QCD":
            pt_min = 300
            pt_max = 1000
        else:
            return "Wrong dataset typ."

        b_jets      = (truth_ == 0) | (truth_ == 1) | (truth_ == 2)
        c_jets      = (truth_ == 3)
        l_jets      = (truth_ == 4) | (truth_ == 5)
        summed_jets = b_jets + c_jets + l_jets

        
        b_pred = output_data_[:, :3].sum(axis=1)
        c_pred = output_data_[:, 3]
        l_pred = output_data_[:, -2:].sum(axis=1)

        
        bvsl = np.where((b_pred + l_pred)>=0, (b_pred)/(b_pred + l_pred), -1)
        cvsb = np.where((b_pred + c_pred)>=0, (c_pred)/(b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred)>=0, (c_pred)/(l_pred + c_pred), -1)
        
        b_veto = ((truth_ != 0) | (truth_ != 1) | (truth_ != 2) | (summed_jets != 0))[(jet_pt_ > pt_min) | (jet_pt_ < pt_max)]
        c_veto = ((truth_ != 3) | (summed_jets != 0))[(jet_pt_ > pt_min) | (jet_pt_ < pt_max)]
        l_veto = ((truth_ != 4) | (truth_ != 5) | (summed_jets != 0))[(jet_pt_ > pt_min) | (jet_pt_ < pt_max)]

        roc_list   = []
        label_list = ["BvsL", "CvsB", "CvsL"]
        roc_list.append(calculate_roc(b_jets, bvsl, c_veto, output_directory, key, label_list[0].lower()))
        roc_list.append(calculate_roc(c_jets, cvsb, l_veto, output_directory, key, label_list[1].lower()))
        roc_list.append(calculate_roc(c_jets, cvsl, b_veto, output_directory, key, label_list[2].lower()))

        plot_roc(roc_list, label_list, key, pt_min, pt_max, output_directory)

# adapted from https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.py and https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.ipynb
def plot_roc(roc_list, label_list, dataset_key, pt_min, pt_max, output_directoy):
    for i, l in enumerate(label_list):
        fpr, tpr, auc = roc_list[i]
        
        plt.figure()
        plt.plot(tpr, fpr, label=f"AUC = {np.round(auc, 3)}", color="blue")
        plt.title(f"{l}, ${pt_min}GeV < p_T < {pt_max}GeV$, {dataset_key} events")
        plt.xlabel("Tagging efficiency")
        plt.ylabel("Mistagging rate")
        plt.yscale("log")
        plt.xlim(0, 1)
        plt.ylim(1e-3, 1)
        plt.grid(which="minor", alpha=0.85)
        plt.grid(which="major", alpha=0.95, color="black")
        plt.legend(loc="best")
        plt.savefig(f"{output_directoy}roc_{dataset_key}_{l.lower()}.pdf")
        plt.close()

# adapted from https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.py and https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.ipynb    
def calculate_roc(truth, dicriminator, veto, output_directory, dataset_key, name):
    fpr, tpr, _ = roc_curve(truth[veto], dicriminator[veto])
        
    index = np.unique(fpr, return_index=True)[1]
    fpr   = np.asarray([fpr[i] for i in sorted(index)])
    tpr   = np.asarray([tpr[i] for i in sorted(index)])
    area  = auc(fpr, tpr)
    np.save(f"{output_directory}roc_{dataset_key}_{name}.npy", np.array([fpr, tpr, area], dtype=object))
    return fpr, tpr, area


def plot_losses(train_loss, test_loss, output_dir):
    plt.title("Losses")
    plt.plot(*np.array(list(enumerate(test_loss, 1))).T, label="Test")
    plt.plot(*np.array(list(enumerate(train_loss, 1))).T, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_dir + "loss.pdf")
    plt.close()
