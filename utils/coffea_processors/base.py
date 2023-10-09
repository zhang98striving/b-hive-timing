import awkward as ak
import os
import hist
import numpy as np
from coffea import processor
from typing import List


class DataPreprocessing_BaseClass(processor.ProcessorABC):
    def __init__(
        self,
        output_directory=None,
        config_dict=None,
        bins_pt: List = None,
        bins_eta: List = None,
        prefix="",
        processes: str = None,
        precision=np.float32,
    ):
        self._accumulator = processor.dict_accumulator({})
        self.bins_eta = bins_eta
        self.bins_pt = bins_pt
        self.config_dict = config_dict
        self.output_dir = output_directory
        self.precision = precision
        self.prefix = prefix
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

        global_arr, cpf_arr, npf_arr, vtx_arr, truth, process = self.callColumnAccumulator(
            output, events, proc_flag
        )

        b_hist.fill(
            global_arr["jet_pt"][truth == 0],
            global_arr["jet_eta"][truth == 0],
        )
        bb_hist.fill(
            global_arr["jet_pt"][truth == 1],
            global_arr["jet_eta"][truth == 1],
        )
        lepb_hist.fill(
            global_arr["jet_pt"][truth == 2],
            global_arr["jet_eta"][truth == 2],
        )
        c_hist.fill(
            global_arr["jet_pt"][truth == 3],
            global_arr["jet_eta"][truth == 3],
        )
        uds_hist.fill(
            global_arr["jet_pt"][truth == 4],
            global_arr["jet_eta"][truth == 4],
        )
        g_hist.fill(
            global_arr["jet_pt"][truth == 5],
            global_arr["jet_eta"][truth == 5],
        )

        output_location = os.path.join(
            self.output_dir, f"{self.prefix}{dataset}_{filename}_{start}_{stop}.npz"
        )

        output_location_list.append(output_location)

        self.saveOutput(output_location, global_arr, cpf_arr, npf_arr, vtx_arr, truth, process)

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
