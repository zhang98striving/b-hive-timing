import os

import numpy as np
import torch
import uproot
from rich.progress import track
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from utils.models.deepjet import DeepJet
from utils.plotting.termplot import terminal_roc
from utils.torch.datasets import DeepJetDataset
from utils.config.config_loader import ConfigLoader
from tasks.training import TrainingTask


class InferenceTask(TrainingDependency, DatasetDependency, BaseTask):
    def requires(self):
        return {"training": TrainingTask.req(self), "dataset": DatasetConstructorTask.req(self)}

    def output(self):
        return {
            "output_root": self.local_target("output.root"),
            "prediction": self.local_target("prediction.npy"),
            "process": self.local_target("process.npy"),
            "truth": self.local_target("truth.npy"),
            "kinematics": self.local_target("kinematics.npy"),
        }

    def run(self):
        os.makedirs(self.local_path(), exist_ok=True)
        config_dict = np.load(self.input()["dataset"]["config_dict"].path, allow_pickle=True).item()

        model = DeepJet(config_dict["model"]["feature_edges"]).to(self.device)
        best_model = torch.load(
            self.input()["training"]["best_model"].path,
            map_location=torch.device(self.device),
        )
        model.load_state_dict(best_model["model_state_dict"])

        config = ConfigLoader.load_config(self.config)

        print("Loading Dataset")
        files = np.array(open(self.input()["dataset"]["file_list"].path, "r").read().split("\n"))
        print("Loading Dataset")
        files = np.array(
            open(self.input()["dataset"]["file_list"].path, "r").read().split("\n")[:-1]
        )
        test_mask = ~(np.char.find(files, "test") == -1)
        test_files = files[test_mask]

        histogram_test = np.load(
            self.input()["dataset"]["histogram_test"].path,
            allow_pickle=True,
        )
        test_data = DeepJetDataset(
            test_files,
            "test",
            histogram_training=histogram_test,
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
        )
        test_data = DeepJetDataset(test_files, "test", histogram_training=histogram_test)
        test_dataloader = DataLoader(test_data, batch_size=10000, num_workers=64)

        model.eval()
        kinematics = []
        truth = []
        process = []
        prediction = []
        output = []
        for x, _, y, proc in track(test_dataloader, "Inference..."):
            x = x.float()

            kinematics.append(x[:, :2, 0])
            truth.append(y)
            process.append(proc)
            with torch.no_grad():
                pred = model(x.to(device=self.device))
                prediction.append(pred)
                if len(output) == 0:
                    output = pred.cpu().numpy()
                else:
                    output = np.append(output, pred.cpu().numpy(), axis=0)

        prediction = torch.cat(prediction, dim=0).cpu().numpy()
        kinematics = torch.cat(kinematics, dim=0).cpu().numpy()
        truth = torch.cat(truth, dim=0).cpu().numpy().astype(int)
        process = torch.cat(process, dim=0).cpu().numpy().astype(int)
        one_hot_truth = np.zeros((len(truth), np.max(truth) + 1))
        one_hot_truth[np.arange(len(truth)), truth] = 1

        np.save(self.output()["kinematics"].path, kinematics)
        np.save(self.output()["prediction"].path, prediction)
        np.save(self.output()["process"].path, process)
        np.save(self.output()["truth"].path, truth)

        terminal_roc(prediction, truth, title="Inference ROC")

        output = np.concatenate((kinematics, prediction, one_hot_truth), axis=1)
        with uproot.recreate(self.output()["output_root"].path) as root_file:
            root_file["tree"] = {
                "Jet_pt": output[:, 0],
                "Jet_eta": output[:, 1],
                "prob_isB": output[:, 2],
                "prob_isBB": output[:, 3],
                "prob_isLeptB": output[:, 4],
                "prob_isC": output[:, 5],
                "prob_isUDS": output[:, 6],
                "prob_isG": output[:, 7],
                "isB": output[:, 8],
                "isBB": output[:, 9],
                "isLeptB": output[:, 10],
                "isC": output[:, 11],
                "isUDS": output[:, 12],
                "isG": output[:, 13],
            }
