import itertools
import os

import luigi
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency
from utils.config.config_loader import ConfigLoader


class InputHistogrammerTask(DatasetDependency, BaseTask):
    config = luigi.Parameter()

    def requires(self):
        return DatasetConstructorTask.req(
            self, dataset_version=self.dataset_version, config=self.config
        )

    def output(self):
        processes = ConfigLoader.load_config(self.config)["processes"]
        attributes = [process + "_hist.npz" for process in processes]
        return {
            key: self.local_target(attr) for (key, attr) in zip(processes, attributes)
        }

    def run(self):
        # Create output directory
        for file in self.output().keys():
            self.output()[file].parent.touch()
        # preparation
        processes = ConfigLoader.load_config(self.config)["processes"]
        files = self.input()["file_list"].load().split("\n")
        Nbins = 20
        histograms = {}
        for proc in processes:
            histograms[proc] = {}
        # TODO: Replace this one with yml parsing
        testfile = np.load(files[0], allow_pickle=True)
        files_in_npz = testfile.files
        dtype_names_f = {}
        for f in files_in_npz:
            dtype_names_f[f] = testfile[f].dtype.names
        # The histogramming step
        # Iterate over datafiles
        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[n]}"),
            expand=True,
        ) as progress:
            task_files = progress.add_task(
                "Processing files... ", total=len(files), n=""
            )
            task_hists = progress.add_task(
                "Making histograms...",
                total=len(files_in_npz) * len(dtype_names_f),
                n="",
            )
            for file in files:
                progress.reset(task_hists)
                progress.update(task_files, n=f"{file.split('/')[-1]}")
                progress.update(task_hists, completed=0, n="")
                newdata = np.load(file, allow_pickle=True)
                for f, (pi, proc) in list(
                    itertools.product(files_in_npz, enumerate(processes))
                ):
                    progress.update(task_hists, n=f"{f}/{proc}")
                    try:
                        dtype = [(n1, "f4") for n1 in dtype_names_f[f]]
                        if f not in histograms[proc].keys():
                            histograms[proc][f] = np.zeros((2, Nbins + 2), dtype=dtype)
                        selection = newdata["process"] == pi
                        if selection.sum() == 0:
                            continue
                        selected_data = newdata[f][selection]
                        for field in dtype_names_f[f]:
                            progress.update(task_hists, n=f"{f}/{proc}:{field}")
                            samples = selected_data[field].ravel()
                            if (histograms[proc][f][field][1] == 0).all():
                                bins = Nbins
                            else:
                                bincenters = histograms[proc][f][field][1]
                                Δbin = bincenters[1] - bincenters[0]
                                bins = bincenters - Δbin / 2
                                bins = np.append(bins, bins[-1] + Δbin)[
                                    1:-1
                                ]  # remove overflow bins
                            N, bins = np.histogram(samples, bins=bins)
                            # add overflow bins
                            N_below = (
                                samples < bins[0]
                            ).sum()  # lower overflow bin content
                            N_above = (
                                samples > bins[-1]
                            ).sum()  # upper overflow bin content
                            N = np.array([N_below, *N, N_above])
                            bins = np.array(
                                [
                                    bins[0] - (bins[1] - bins[0]),  # lower overflow bin
                                    *bins,
                                    bins[-1]
                                    + (bins[1] - bins[0]),  # upper overflow bin
                                ]
                            )
                            bincenters = 0.5 * (bins[:-1] + bins[1:])
                            histograms[proc][f][field][0] += N
                            histograms[proc][f][field][1] = bincenters.astype(
                                np.float32
                            )
                            progress.update(task_hists, advance=1)
                    except Exception as error:
                        # if data is empty:
                        if self.debug:
                            if newdata[f].dtype.names == ():
                                print("'()' exception in", f, file, proc)
                                continue
                        # if data has a single array in it, dtype is None (eg. process, weights):
                        if newdata[f].dtype.names == None:
                            if self.debug:
                                print("'None' exception in", f, file, proc)
                            continue
                        else:
                            raise error
                for process in processes:
                    np.savez(self.output()[process].path, **histograms[process])
                progress.update(
                    task_hists, completed=len(files_in_npz) * len(dtype_names_f)
                )
                progress.update(task_files, advance=1)


class HistogramPlotterTask(DatasetDependency, BaseTask):
    dataset_version = luigi.Parameter()
    config = luigi.Parameter()
    ylog = luigi.BoolParameter(default=False)

    def requires(self):
        return InputHistogrammerTask.req(
            self, dataset_version=self.dataset_version, config=self.config
        )

    def output(self):
        processes = ConfigLoader.load_config(self.config)["processes"]
        # FIXME fix loading if the previous task has no outputs yet
        files = np.load(self.input()[list(self.input().keys())[0]].path).files
        return_dict = {}
        for process, file in itertools.product(processes, files):
            return_path = os.path.join(process, file, "done")
            return_dict[process + "_" + file] = self.local_target(
                return_path + ("_ylog" if self.ylog else "")
            )
        return return_dict

    def run(self):
        for file in self.output().keys():
            self.output()[file].parent.touch()
        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[n]}"),
            expand=True,
        ) as progress:
            task_process = progress.add_task(
                f"Processing histograms...",
                total=len(self.input().items()),
                n="",
            )
            task_features = progress.add_task(
                f"Processing features... ",
                n="",
            )
            task_plots = progress.add_task(
                "Plotting histograms...",
                n="",
            )
            for process, file in self.input().items():
                histograms = np.load(file.path)
                progress.reset(task_features)
                progress.reset(task_plots)
                progress.update(
                    task_process,
                    description=f"Processing histograms for process {process:15}",
                    n=process,
                )
                for attribute in histograms.files:
                    progress.update(
                        task_features,
                        total=len(histograms.files),
                        n=f"{attribute[:17]+('...' if len(attribute) >= 17 else ''):20}",
                        description=f"Processing features in {attribute}",
                    )
                    progress.update(
                        task_plots,
                        completed=0,
                        total=len(histograms[attribute].dtype.name),
                    )
                    for variable in histograms[attribute].dtype.names:
                        progress.update(
                            task_plots,
                            total=len(histograms[attribute].dtype.names),
                            n=f"{variable[:17]+('...' if len(variable) >= 17 else ''):20}",
                        )
                        N, bincenters = histograms[attribute][variable]
                        dx = bincenters[1] - bincenters[0]
                        plt.bar(bincenters, N, width=dx, color="C0")
                        plt.title("Plot of " + variable + " for " + process)
                        plt.xlabel(variable)
                        plt.ylabel("N")
                        if self.ylog:
                            plt.yscale("log")
                        plt.savefig(
                            self.output()[process + "_" + attribute].dirname
                            + "/"
                            + variable
                            + ("_log" if self.ylog else "")
                            + ".png"
                        )
                        plt.close()
                        progress.update(task_plots, advance=1)
                    progress.update(
                        task_features,
                        advance=1,
                    )
                    progress.update(
                        task_plots, completed=len(histograms[attribute].dtype.names)
                    )
                progress.update(task_process, advance=1)
            for key, file in self.output().items():
                file.touch()
