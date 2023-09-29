import awkward as ak
import hist
import numpy as np
from coffea import processor
from typing import List


class DataPreprocessing_BaseClass(processor.ProcessorABC):
    def __init__(
        self,
        output_directory,
        config_dict,
        bins_pt: List=None,
        bins_eta: List=None,
        prefix="",
        processes: str = None,
    ):
        self.prefix = prefix
        self.config_dict = config_dict
        self.output_dir = output_directory
        self._accumulator = processor.dict_accumulator({})
        self.lower_pt = 10
        self.upper_pt = 2000
        self.lower_eta = -4.0
        self.upper_eta = 4.0
        self.bins_pt = bins_pt
        self.bins_eta = bins_eta
        self.processes = processes
        if self.processes is None:
            self.processes = []

        self.b_hist = (
            hist.Hist.new.Variable(self.bins_pt, name="pt")
            .Variable(self.bins_eta, name="eta")
            .Int64()
        )
        self.bb_hist = (
            hist.Hist.new.Variable(self.bins_pt, name="pt")
            .Variable(self.bins_eta, name="eta")
            .Int64()
        )
        self.lepb_hist = (
            hist.Hist.new.Variable(self.bins_pt, name="pt")
            .Variable(self.bins_eta, name="eta")
            .Int64()
        )
        self.c_hist = (
            hist.Hist.new.Variable(self.bins_pt, name="pt")
            .Variable(self.bins_eta, name="eta")
            .Int64()
        )
        self.uds_hist = (
            hist.Hist.new.Variable(self.bins_pt, name="pt")
            .Variable(self.bins_eta, name="eta")
            .Int64()
        )
        self.g_hist = (
            hist.Hist.new.Variable(self.bins_pt, name="pt")
            .Variable(self.bins_eta, name="eta")
            .Int64()
        )
        self.setFeatureNamesAndEdges()

    def setFeatureNamesAndEdges(self):
        pass

    def saveOutput(self, output_location, output):
        pass

    @property
    def accumulator(self):
        return self._accumulator

    def callColumnAccumulator(self, output, events):
        pass

    def process(self, events):
        dataset = events.metadata["dataset"]

        start = events.metadata["entrystart"]
        stop = events.metadata["entrystop"]
        filename = "_".join(events.metadata["filename"].split("/")[1:]).split(".")[0]

        # assign process number
        proc_flag = -1
        for (
            i,
            proc,
        ) in enumerate(self.processes):
            if proc in dataset:
                proc_flag = i

        output = self.accumulator
        output_location_list = []

        b_hist = self.b_hist
        bb_hist = self.bb_hist
        lepb_hist = self.lepb_hist
        c_hist = self.c_hist
        uds_hist = self.uds_hist
        g_hist = self.g_hist

        output = self.callColumnAccumulator(output, events, proc_flag)

        b_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_truth"].value == 0],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_truth"].value == 0],
        )
        bb_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_truth"].value == 1],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_truth"].value == 1],
        )
        lepb_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_truth"].value == 2],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_truth"].value == 2],
        )
        c_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_truth"].value == 3],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_truth"].value == 3],
        )
        uds_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_truth"].value == 4],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_truth"].value == 4],
        )
        g_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_truth"].value == 5],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_truth"].value == 5],
        )

        output_location = (
            f"{self.output_dir}".rstrip(" / ")
            + f"/{self.prefix}{dataset}_{filename}_{start}_{stop}.npy"
        )
        output_location_list.append(output_location)

        self.saveOutput(output_location, output)

        return {
            "output_location": output_location_list,
            "b_hist": np.sum([b_hist.view()], axis=0),
            "bb_hist": np.sum([bb_hist.view()], axis=0),
            "lepb_hist": np.sum([lepb_hist.view()], axis=0),
            "c_hist": np.sum([c_hist.view()], axis=0),
            "uds_hist": np.sum([uds_hist.view()], axis=0),
            "g_hist": np.sum([g_hist.view()], axis=0),
        }

    def postprocess(self, accumulator):
        pass

