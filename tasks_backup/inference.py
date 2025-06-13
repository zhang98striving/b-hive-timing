from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from tasks.training import TrainingTask
from torch.utils.data import DataLoader
from utils.adversarial_attacks.pick_attack import pick_attack
from utils.config.config_loader import ConfigLoader
from utils.models.models import BTaggingModels
from utils.plotting.termplot import terminal_roc
import numpy as np
import torch
import law


from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import (
    AttackDependency,
    DatasetDependency,
    TrainingDependency,
    TestAttackDependency,
    TestDatasetDependency,
)
from tasks.training import TrainingTask
from utils.config.config_loader import ConfigLoader
from utils.models.models import BTaggingModels

# to make formatters work
law.contrib.load("numpy")


class InferenceTask(
    TestAttackDependency, AttackDependency, TrainingDependency, TestDatasetDependency, DatasetDependency, BaseTask
):
    def requires(self):
        return {
            "training": TrainingTask.req(self), # this is to make cli-steering with different attack possible
            "test_dataset": DatasetConstructorTask.req(
                self,
                dataset_version=self.test_dataset_version,
                filelist=self.test_filelist,
            ),
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
        if issubclass(type(BTaggingModels(self.model_name)), torch.nn.Module):
            model = BTaggingModels(self.model_name).to(self.device)
            best_model = torch.load(
                self.input()["training"]["best_model"].path,
                map_location=torch.device(self.device),
            )
            model.load_state_dict(best_model["model_state_dict"])
        else:
            model = BTaggingModels(self.model_name)
            model.model = keras.models.load_model(
                self.input()["training"]["best_model"].path,
                custom_objects = model.custom_objects
            )

        # Picking attack
        print(
            rf"Will apply {self.test_attack} attack with epsilon={self.test_attack_magnitude} and {self.test_attack_iterations} iterations."
        )
        attack = pick_attack(
            self.test_attack,
            device=self.device,
            integer_positions=model.integers,
            default_values=model.defaults,
            epsilon=self.test_attack_magnitude,
            epsilon_factors=self.test_attack_individual_factors,
            iterations=self.test_attack_iterations,
            reduce=self.test_attack_reduce,
            restrict_impact=self.test_attack_restrict_impact,
        )

        print("Loading Dataset")
        files = self.input()["test_dataset"]["file_list"].load().split("\n")

        histogram_test = self.input()["test_dataset"]["histogram"].load(
            formatter="numpy", allow_pickle=True
        )

        print("Initialize datasets")
        datasetClass = model.datasetClass
        test_data = datasetClass(
            files,
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
            pin_memory=True,  # Pin Memory for faster CPU/GPU memory load
        )

        test_dataloader.nits_expected = len(test_dataloader)

        print("Start inference")
        predictions, truths, kinematics, processes = model.predict_model(
            test_dataloader, self.device, attack=attack
        )

        np.save(self.output()["kinematics"].path, kinematics)
        np.save(self.output()["prediction"].path, predictions)
        np.save(self.output()["process"].path, processes)
        np.save(self.output()["truth"].path, truths)

        terminal_roc(predictions, truths, title="Inference ROC")
