import os
import law

import numpy as np
import torch
import uproot
from rich.progress import track
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from tasks.training import TrainingTask
from utils.config.config_loader import ConfigLoader
from utils.models.models import BTaggingModels
from utils.plotting.termplot import terminal_roc
from utils.torch import DeepJetDataset

# to make formatters work
law.contrib.load("numpy")


class InferenceTask(TrainingDependency, DatasetDependency, BaseTask):
    def requires(self):
        return {
            "training": TrainingTask.req(self),
            "dataset": DatasetConstructorTask.req(self),
        }

    def output(self):
        return {
            # "output_root": self.local_target("output.root"),
            "prediction": self.local_target("prediction.npy"),
            "process": self.local_target("process.npy"),
            "truth": self.local_target("truth.npy"),
            "kinematics": self.local_target("kinematics.npy"),
        }

    def run(self):
        # create directory
        self.output()["prediction"].parent.touch()
        config = ConfigLoader.load_config(self.config)

        # Model Defintion
        print("Build Model")
        print(self.model_name)
        model = BTaggingModels(self.model_name).to(self.device)
        best_model = torch.load(
            self.input()["training"]["best_model"].path,
            map_location=torch.device(self.device),
        )
        model.load_state_dict(best_model["model_state_dict"])

        print("Loading Dataset")
        files = np.array(
            open(self.input()["dataset"]["file_list"].path, "r").read().split("\n")[:-1]
        )
        test_mask = ~(np.char.find(files, "test") == -1)
        test_files = files[test_mask]

        histogram_test = self.input()["dataset"]["histogram_test"].load(
            formatter="numpy", allow_pickle=True
        )

        print("Initialize datasets")
        datasetClass = model.datasetClass
        test_data = datasetClass(
            test_files,
            model,
            data_type="test",
            histogram_training=histogram_test,
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            num_workers=self.n_threads,
        )

        print("Start inference")
        predictions, truths, kinematics, processes = model.predict(
            test_dataloader, self.device
        )

        one_hot_truth = np.zeros((len(truths), np.max(truths) + 1))
        one_hot_truth[np.arange(len(truths)), truths] = 1

        np.save(self.output()["kinematics"].path, kinematics)
        np.save(self.output()["prediction"].path, predictions)
        np.save(self.output()["process"].path, processes)
        np.save(self.output()["truth"].path, truths)

        terminal_roc(predictions, truths, title="Inference ROC")

        # joined_output = np.concatenate((kinematics, predictions, one_hot_truth), axis=1)
        # with uproot.recreate(self.output()["output_root"].path) as root_file:
        #     root_file["tree"] = {
        #         "Jet_pt": joined_output[:, 0],
        #         "Jet_eta": joined_output[:, 1],
        #         "prob_isB": joined_output[:, 2],
        #         "prob_isBB": joined_output[:, 3],
        #         "prob_isLeptB": joined_output[:, 4],
        #         "prob_isC": joined_output[:, 5],
        #         "prob_isUDS": joined_output[:, 6],
        #         "prob_isG": joined_output[:, 7],
        #         "isB": joined_output[:, 8],
        #         "isBB": joined_output[:, 9],
        #         "isLeptB": joined_output[:, 10],
        #         "isC": joined_output[:, 11],
        #         "isUDS": joined_output[:, 12],
        #         "isG": joined_output[:, 13],
        #     }
