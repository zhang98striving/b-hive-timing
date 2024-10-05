import os
import random
from typing import Dict
from collections import defaultdict
import uuid
import hist
import luigi
import numpy as np
from coffea import processor
from coffea.nanoevents import BaseSchema
from rich.progress import track

from tasks.base import BaseTask
from tasks.parameter_mixins import DatasetDependency
from utils.coffea_processors.processor_loader import ProcessorLoader
from utils.config.config_loader import ConfigLoader
from utils.dataset.merging import merge_datasets
from utils.weighting.histogram import (
    histogram_weighting,
    weight_all_files_histrogram_weighting,
)


def read_in_samples_match_processes(file_path, processes):
    samples_dict = defaultdict(list)
    with open(file_path, "r") as input_txt:
        while line := input_txt.readline().rstrip("\n"):
            if line:
                if processes == ["default"]:
                    samples_dict[uuid.uuid4().hex].append(line)
                else:
                    for process in processes:
                        if process in line:
                            samples_dict[process].append(line)
                            break
    return samples_dict


class DatasetConstructorTask(DatasetDependency, BaseTask):
    coffea_worker = luigi.IntParameter(
        default=1,
        description="Number of workers for Coffea-processing",
        significant=False,
    )

    chunk_size = luigi.IntParameter(
        default=100000, description="Number of events for outgoing files"
    )

    def output(self):
        return {
            "file_list": self.local_target("processed_files.txt"),
            "histogram": self.local_target("histogram.npy"),
        }

    def run(self):
        print("Dataset construction")
        self.output()["file_list"].parent.touch()  # create directory
        config = ConfigLoader.load_config(self.config)
        np.random.seed(self.seed)
        assert self.filelist != "", (
            "You did not specify a filelist .txt but tried to run a new DatasetConstruction! "
            "Either you forgot to specify the path to the file or are using a wrong dataset-version!"
        )


        all_files = []
        samples = read_in_samples_match_processes(self.filelist, config["processes"])

        if self.debug:
            # skim files to only 10 files per process
            samples = {
                key: values[0 : min(len(samples), 1)] for key, values in samples.items()
            }

        # Make a dictionary entry for all of them:

        futures_run = processor.Runner(
            executor=processor.FuturesExecutor(
                compression=None, workers=self.coffea_worker
            ),
            schema=BaseSchema,
            chunksize=self.chunk_size//20, # should be << chunk_size in order to get everything shuffled correctly
            maxchunks=None if not (self.debug) else 10,
        )
        processorClass = ProcessorLoader(config.get("processor", "PFCandidateAndVertexProcessing"),  
                output_directory=self.local_path(),
                bins_pt=config.get("bins_pt", None),
                bins_eta=config.get("bins_eta", None),
                processes=config.get("processes", None),
                global_features=config.get("global_features", []),
                cpf_candidates=config.get("cpf_candidates", []),
                npf_candidates=config.get("npf_candidates", []),
                vtx_features=config.get("vtx_features", []),
                n_cpf_candidates=config.get("n_cpf_candidates", 50),
                n_npf_candidates=config.get("n_npf_candidates", 50),
                n_vtx_features=config.get("n_vtx_features", 5),
                truths=config.get("truths", None),
                )
        print("Processor:")
        print(processorClass)

        output = futures_run(
            samples,
            treename=config["treename"],
            processor_instance=processorClass
        )

        # saving histograms from coffea
        histograms = []
        file_list = []
        for key, value in output.items():
            if key == "output_location":
                file_list += value  # append output location list
            else:
                histograms.append(value.view())  # append view on hist -> np.array

        training_histograms = {
            key: value for key, value in output.items() if not "output_location" in key
        }

        np.save(
            self.output()[f"histogram"].path,
            np.array(histograms, dtype=np.float32),
        )

        print("Start merging files")
        # returns list of merged training-files
        all_files += merge_datasets(
            file_list,
            self.local_path(),
            label="file",
            processor=config.get("processor", "PFCandidateAndVertexProcessing"),
            chunk_size=self.chunk_size,
            shuffle=True,
            histograms=np.array(histograms, dtype=np.float32),
            reference_key=config["truths"].index(config["reference_flavour"]), #reference_key is the histogram index for LZ4 dataset
            bins_pt=config["bins_pt"],
            bins_eta=config["bins_eta"],
        )
        if "LZ4" not in config.get("processor", "PFCandidateAndVertexProcessing"):
            # delete unmerged files
            for file in file_list:
                os.remove(file)

            # add weights to all files
            # this should be done on the fly - please implement!
            all_files = weight_all_files_histrogram_weighting(
                all_files,
                histograms=training_histograms,
                bins_pt=config["bins_pt"],
                bins_eta=config["bins_eta"],
                reference_key=config["reference_flavour"],
            )

        self.output()["file_list"].dump("\n".join(all_files), formatter="text")
