import os
from typing import List

import awkward as ak
import hist
import numpy as np
from coffea import processor


class DataPreprocessing_BaseClass(processor.ProcessorABC):
    def __init__(
        self,
        output_directory=None,
        bins_pt: List = None,
        bins_eta: List = None,
        prefix="",
        precision=np.float32,
        global_features: List[str] = None,
        cpf_candidates: List[str] = None,
        npf_candidates: List[str] = None,
        vtx_features: List[str] = None,
        truths: List[str] = None,
        processes: List[str] = None,
    ):
        self._accumulator = processor.dict_accumulator({})
        self.bins_eta = bins_eta
        self.bins_pt = bins_pt
        self.output_dir = output_directory
        self.precision = precision
        self.prefix = prefix
        self.processes = processes
        self.cpf = cpf_candidates
        self.npf = npf_candidates
        self.vtx = vtx_features
        self.global_features = global_features
        self.truths = truths
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
        feature_edges = []
        feature_names = []

        feature_names.append(self.global_features)
        feature_edges.append(len(feature_names))
        feature_edges.append(feature_edges[-1] + len(self.cpf) * self.n_cpf)
        feature_names.extend(self.cpf)
        feature_edges.append(feature_edges[-1] + len(self.npf) * self.n_npf)
        feature_names.extend(self.npf)
        feature_edges.append(feature_edges[-1] + len(self.vtx) * self.n_vtx)
        feature_names.extend(self.vtx)
        feature_names.append("truths")
        feature_names.extend(self.truths)
        feature_names.append("process")

        self.feature_edges = feature_edges
        self.features = feature_names

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

        (
            global_arr,
            cpf_arr,
            npf_arr,
            vtx_arr,
            truth,
            process,
        ) = self.callColumnAccumulator(output, events, proc_flag)

        b_hist.fill(
            global_arr["jet_pt"][truth["isB"]],
            global_arr["jet_eta"][truth["isB"]],
        )
        bb_hist.fill(
            global_arr["jet_pt"][truth["isBB"]],
            global_arr["jet_eta"][truth["isBB"]],
        )
        lepb_hist.fill(
            global_arr["jet_pt"][truth["isLeptonicB"]],
            global_arr["jet_eta"][truth["isLeptonicB"]],
        )
        c_hist.fill(
            global_arr["jet_pt"][truth["isC"]],
            global_arr["jet_eta"][truth["isC"]],
        )
        uds_hist.fill(
            global_arr["jet_pt"][truth["isUD"] & truth["isS"]],
            global_arr["jet_eta"][truth["isUD"] & truth["isS"]],
        )
        g_hist.fill(
            global_arr["jet_pt"][truth["isG"]],
            global_arr["jet_eta"][truth["isG"]],
        )

        output_location = os.path.join(
            self.output_dir, f"{self.prefix}{dataset}_{filename}_{start}_{stop}.npz"
        )

        output_location_list.append(output_location)

        self.saveOutput(
            output_location, global_arr, cpf_arr, npf_arr, vtx_arr, truth, process
        )

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
