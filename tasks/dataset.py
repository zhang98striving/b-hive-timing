from coffea.nanoevents import BaseSchema, PFNanoAODSchema
from tasks.base import BaseTask, config_dict
from tasks.parameter_mixins import DatasetDependency
from coffea.nanoevents.methods import base
from rich.progress import track
from utils.processors import DeepJet_NTupleDataPreprocessing
from coffea import processor
import os
import numpy as np
import traceback
import luigi


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

    def output(self):
        return {
            "file_list": self.local_target("processed_files.txt"),
            "config_dict": self.local_target("config.npy"),
            "histogram_training": self.local_target("histogram_training.npy"),
            "histogram_test": self.local_target("histogram_test.npy"),
        }

    def run(self):
        print("Dataset construction")
        os.makedirs(self.local_path(), exist_ok=True)
        output_string = ""
        np.random.seed(1)
        for sample_prefix in ["training", "test"]:
            path = self.training_dataset_path if "training" else self.test_dataset_path
            samples = open(path, "r").read().split("\n")[:-1]

            if self.debug:
                samples = samples[0 : min(len(samples), 100)]

            # Get all dataset name prefixes:
            l = []
            for ti in samples:
                dataset_name = ti.split("_TuneCP5")[0].split("/")[-1]
                if dataset_name not in l:
                    l.append(dataset_name)

            # Make a dictionary entry for all of them:
            sample_dict = {}
            for li in l:
                mask = np.core.defchararray.find(samples, li) != -1
                sample_dict[sample_prefix + "_" + li] = np.array(samples)[mask].tolist()

            futures_run = processor.Runner(
                executor=processor.FuturesExecutor(compression=None, workers=self.coffea_worker),
                schema=BaseSchema,
                chunksize=10000,
                maxchunks=None if not (self.debug) else 10,
            )
            output = futures_run(
                sample_dict,
                "DeepJetNTupler/DeepJetvars",
                processor_instance=DeepJet_NTupleDataPreprocessing(
                    self.local_path(), config_dict, ""
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
                    output_location = f"{self.local_path()}/{key}.npy"
                    np.save(output_location, output[key])
                    histograms.append(output[key])
            np.save(
                self.output()[f"histogram_{sample_prefix}"].path,
                np.array(histograms),
            )

            print(f"number of output {sample_prefix} files:", len(file_list))

            np.save(self.output()["config_dict"].path, config_dict)
            files = np.array(file_list)
            np.random.shuffle(files)
            chunk_size = 100000
            dim = np.load(files[0], allow_pickle=True).shape[-1]
            chunk = np.empty((chunk_size, dim))
            N_tot = np.load(
                self.output()[f"histogram_{sample_prefix}"].path,
            ).sum()
            i = 0
            j = 0
            Ns = 0
            n_chunk = 0
            # Merge datasets for the training set:
            if sample_prefix == "training":
                N_s = [N_tot]
                labels = ["training"]
                for file in track(files, "Merging..."):
                    data = np.load(file, allow_pickle=True)
                    n_samples = data.shape[0]
                    # chunk overflow:
                    if n_chunk + n_samples > chunk_size:
                        index_range = chunk_size - n_chunk
                    else:
                        index_range = n_samples
                    # Category overflow:
                    if n_samples + Ns > N_s[i]:
                        index_range = N_s[i] - Ns
                    Ns += index_range
                    chunk[n_chunk : n_chunk + index_range] = data[:index_range]
                    n_chunk += index_range
                    if n_chunk == chunk_size or Ns == N_s[i]:
                        filename = f"{self.local_path()}/{labels[i]}_{j}.npy"
                        np.save(filename, chunk[:n_chunk])
                        output_string += f"{filename}\n"
                        j += 1
                        chunk = np.zeros((chunk_size, dim))
                        chunk[: n_samples - index_range] = data[index_range:]
                        n_chunk = n_samples - index_range
                        if Ns == N_s[i]:
                            i += 1
                            Ns = 0
                            j = 0
                        Ns += n_samples - index_range
            else:
                N_s = [int(0.5 * N_tot), N_tot - int(0.5 * N_tot)]
                labels = ["test", "validation"]
                data_origin = ""
                for file in track(files, "Merging..."):
                    data = np.load(file, allow_pickle=True)
                    n_samples = data.shape[0]
                    # chunk overflow:
                    if n_chunk + n_samples > chunk_size:
                        index_range = chunk_size - n_chunk
                    else:
                        index_range = n_samples
                    # Category overflow:
                    if n_samples + Ns > N_s[i]:
                        index_range = N_s[i] - Ns
                    Ns += index_range
                    chunk[n_chunk : n_chunk + index_range] = data[:index_range]
                    data_origin += f"{file}\n" * index_range
                    n_chunk += index_range
                    if n_chunk == chunk_size or Ns == N_s[i]:
                        filename = f"{self.local_path()}/{labels[i]}_{j}.npy"
                        print(
                            data_origin,
                            file=open(".".join(filename.split(".")[:-1]) + ".txt", "w"),
                        )
                        data_origin = f"{file}\n" * (n_samples - index_range)
                        np.save(filename, chunk[:n_chunk])
                        output_string += f"{filename}\n"
                        j += 1
                        chunk = np.empty((chunk_size, dim))
                        chunk[: n_samples - index_range] = data[index_range:]
                        n_chunk = n_samples - index_range
                        if Ns == N_s[i]:
                            i += 1
                            Ns = 0
                            j = 0
                        Ns += n_samples - index_range

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
        bins_pt = [
            10,
            25,
            30,
            35,
            40,
            45,
            50,
            60,
            75,
            100,
            125,
            150,
            175,
            200,
            250,
            300,
            400,
            500,
            600,
            2001,
        ]
        bins_eta = [-4.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1, 1.5, 2.0, 2.5, 4.1]
        for file in track(output_string.split("\n")[:-1], "Evaluating and saving the weights..."):
            samples = np.load(file)
            pt_coordinate = np.digitize(samples[:, 0], bins_pt) - 1
            eta_coordinate = np.digitize(samples[:, 1], bins_eta) - 1
            w = np.array(weights_list)[
                np.array(samples[:, -1], dtype=int), pt_coordinate, eta_coordinate
            ]
            samples = np.insert(samples, -1, w, axis=1)
            np.save(file, samples)

        self.output()["file_list"].dump(f"{output_string}", formatter="text")


def empty_column_accumulator():
    return processor.column_accumulator(np.array([], dtype=np.float64))


def array_accumulator():
    return processor.defaultdict_accumulator(empty_column_accumulator)
