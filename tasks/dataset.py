import os
import random

import luigi
import numpy as np
from coffea import processor
from coffea.nanoevents import BaseSchema
from rich.progress import track

from tasks.base import BaseTask
from tasks.parameter_mixins import DatasetDependency
from utils.coffea_processors.hlt import HLTDataPreprocessing
from utils.config.config_loader import ConfigLoader
from utils.dataset.merging import merge_datasets


class DatasetConstructorTask(DatasetDependency, BaseTask):
    training_dataset_path = luigi.Parameter(
        default=None, description="txt file with input root files for training."
    )
    test_dataset_path = luigi.Parameter(
        default=None, description="txt file with input root files for testing."
    )

    coffea_worker = luigi.IntParameter(
        default=1, description="Number of workers for Coffea-processing"
    )

    chunk_size = luigi.IntParameter(
        default=1000000, description="Number of events for outgoing files"
    )

    test_val_split = 0.5

    def output(self):
        return {
            "file_list": self.local_target("processed_files.txt"),
            "histogram_training": self.local_target("histogram_training.npy"),
            "histogram_test": self.local_target("histogram_test.npy"),
        }

    def run(self):
        print("Dataset construction")
        os.makedirs(self.local_path(), exist_ok=True)
        config = ConfigLoader.load_config(self.config)
        all_files = []
        np.random.seed(1)
        for sample_prefix in ["training", "test"]:
            path = (
                self.training_dataset_path
                if (sample_prefix == "training")
                else self.test_dataset_path
            )
            samples = open(path, "r").read().split("\n")[:-1]

            if self.debug:
                samples = samples[0 : min(len(samples), 10)]

            # Get all dataset name prefixes:
            l = []
            for ti in samples:
                dataset_name = ti.split("_TuneCP5")[0].split("/")[-1]
                if dataset_name not in l:
                    l.append(dataset_name)

            # Make a dictionary entry for all of them:
            sample_dict = {}
            print(f"working on {path}")
            for li in l:
                mask = np.core.defchararray.find(samples, li) != -1
                sample_dict[sample_prefix + "_" + li] = np.array(samples)[mask].tolist()

            futures_run = processor.Runner(
                executor=processor.FuturesExecutor(compression=None, workers=self.coffea_worker),
                schema=BaseSchema,
                chunksize=self.chunk_size,
                maxchunks=None if not (self.debug) else 10,
            )
            output = futures_run(
                sample_dict,
                treename=config["treename"],
                processor_instance=HLTDataPreprocessing(
                    output_directory=self.local_path(),
                    bins_pt=config["bins_pt"],
                    bins_eta=config["bins_eta"],
                    processes=config["processes"],
                ),
            )

            # saving histograms from coffea
            histograms = []
            file_list = []
            for key in output.keys():
                if key == "output_location":
                    for line in output["output_location"]:
                        file_list.append(f"{line}")
                else:
                    histograms.append(output[key])
            np.save(
                self.output()[f"histogram_{sample_prefix}"].path,
                np.array(histograms, dtype=np.float32),
            )

            print(f"number of output {sample_prefix} files:", len(file_list))

            random.shuffle(file_list)

            if sample_prefix == "training":
                # returns list of merged training-files
                all_files += merge_datasets(
                    file_list, self.local_path(), label="train", chunk_size=self.chunk_size
                )
            else:
                n_files_test = int(len(file_list) * self.test_val_split)
                files_test = file_list[:n_files_test]
                files_val = file_list[n_files_test:]
                # returns list of merged test-files
                all_files += merge_datasets(
                    files_test, self.local_path(), label="test", chunk_size=self.chunk_size
                )
                # returns list of merged validation-files
                all_files += merge_datasets(
                    files_val,
                    self.local_path(),
                    label="validation",
                    chunk_size=self.chunk_size,
                )
            # delete unmerged files
            for file in file_list:
                os.remove(file)

        # Get the weights
        histograms = np.load(
            self.output()["histogram_training"].path,
            allow_pickle=True,
        )
        reference_histogram = histograms[0]
        reference_histogram = reference_histogram / np.max(reference_histogram)
        weights_list = []
        for c in range(6):
            other_histogram = histograms[c]
            other_histogram = other_histogram / np.max(other_histogram)
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where(other_histogram > 0, reference_histogram / other_histogram, -10)
            weights = weights / np.max(weights)

            weights[weights < 0] = 1
            weights[weights == np.nan] = 1

            weights_list.append(weights)
        for file in track(all_files, "Evaluating and saving the weights..."):
            samples = np.load(file, allow_pickle=True)
            pt_coordinate = np.digitize(samples["global_features"]["jet_pt"], config["bins_pt"]) - 1
            eta_coordinate = (
                np.digitize(samples["global_features"]["jet_eta"], config["bins_eta"]) - 1
            )

            w = np.array(weights_list)[
                np.array(samples["truth"], dtype=int), pt_coordinate, eta_coordinate
            ]
            np.savez(file, **samples, weight=w)

        self.output()["file_list"].dump("\n".join(all_files), formatter="text")
