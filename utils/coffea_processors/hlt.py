from typing import List

import awkward as ak
import hist
import numpy as np
from coffea import processor
from functools import reduce

from utils.coffea_processors.base import DataPreprocessing_BaseClass
from utils.dataset.structured_arrays import structured_array_from_tree


class HLTDataPreprocessing(DataPreprocessing_BaseClass):
    n_cpf = 26
    n_npf = 25
    n_vtx = 5

    def callColumnAccumulator(self, output, events, flag):
        # slicing based on p_T and eta
        pt_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) >= min(self.bins_pt),
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) <= max(self.bins_pt),
        )
        eta_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) >= min(self.bins_eta),
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) <= max(self.bins_eta),
        )

        # isB = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isB"], axis=0), self.precision)
        # )
        # isBB = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isBB"], axis=0), self.precision)
        # )
        # isGBB = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isGBB"], axis=0), self.precision)
        # )
        # isLeptonicB = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isLeptonicB"], axis=0), self.precision)
        # )
        # isLeptonicB_C = ak.to_numpy(
        #     ak.values_astype(
        #         ak.flatten(events["isLeptonicB_C"], axis=0), self.precision
        #     )
        # )
        # isC = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isC"], axis=0), self.precision)
        # )
        # isCC = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isCC"], axis=0), self.precision)
        # )
        # isGCC = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isGCC"], axis=0), self.precision)
        # )
        # isUD = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isUD"], axis=0), self.precision)
        # )
        # isS = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isS"], axis=0), self.precision)
        # )
        # isG = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isG"], axis=0), self.precision)
        # )
        # isUndefined = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isUndefined"], axis=0), self.precision)
        # )
        # isTau = ak.to_numpy(
        #     ak.values_astype(ak.flatten(events["isTau"], axis=0), self.precision)
        # )
        data_slice = np.array(
            (pt_slice & eta_slice)
            & reduce(
                np.logical_or,
                [
                    ak.to_numpy(ak.flatten(events[truth], axis=0))
                    for truth in self.truths
                ],
            ),
            dtype=bool,
        )
        # data_slice = np.array(
        #     (pt_slice & eta_slice)
        #     & (
        #         np.ndarray.astype(isB, self.precision)
        #         | np.ndarray.astype(isBB, self.precision)
        #         | np.ndarray.astype(isGBB, self.precision)
        #         | np.ndarray.astype(isLeptonicB, self.precision)
        #         | np.ndarray.astype(isLeptonicB_C, self.precision)
        #         | np.ndarray.astype(isC, self.precision)
        #         | np.ndarray.astype(isCC, self.precision)
        #         | np.ndarray.astype(isGCC, self.precision)
        #         | np.ndarray.astype(isUD, self.precision)
        #         | np.ndarray.astype(isS, self.precision)
        #         | np.ndarray.astype(isG, self.precision)
        #     )
        #     & np.logical_not(np.ndarray.astype(isUndefined, self.precision))
        #     & np.logical_not(np.ndarray.astype(isTau, self.precision)),
        #     dtype=bool,
        # )

        global_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.global_features,
            precision=self.precision,
            feature_length=1,
        )

        cpf_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.cpf,
            precision=self.precision,
            feature_length=self.n_cpf,
        )
        npf_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.npf,
            precision=self.precision,
            feature_length=self.n_npf,
        )
        vtx_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.vtx,
            precision=self.precision,
            feature_length=self.n_vtx,
        )
        truth_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.truths,
            precision=np.bool8,
            feature_length=1,
        )

        # target_class = np.full_like(isB, -999)
        # target_class = np.where(isB == 1, 0, target_class)  # b
        # target_class = np.where((isBB == 1) | (isGBB == 1), 1, target_class)  # bb
        # target_class = np.where(
        #     (isLeptonicB == 1) | (isLeptonicB_C == 1), 2, target_class
        # )  # leptonicb
        # target_class = np.where(
        #     (isC == 1) | (isCC == 1) | (isGCC == 1), 3, target_class
        # )  # c
        # target_class = np.where((isUD == 1) | (isS == 1), 4, target_class)  # uds
        # target_class = np.where(isG == 1, 5, target_class)  # g

        # truth = target_class[data_slice]

        # create an array with the process value
        process = np.full_like(global_arr, flag)

        glob_mask = reduce(
            np.logical_and,
            [np.any(~np.isnan(global_arr[key])) for key in global_arr.dtype.names],
        )
        cpf_mask = reduce(
            np.logical_and,
            [np.any(~np.isnan(cpf_arr[key]), axis=1) for key in cpf_arr.dtype.names],
        )
        npf_mask = reduce(
            np.logical_and,
            [np.any(~np.isnan(npf_arr[key]), axis=1) for key in npf_arr.dtype.names],
        )
        vtx_mask = reduce(
            np.logical_and,
            [np.any(~np.isnan(vtx_arr[key]), axis=1) for key in vtx_arr.dtype.names],
        )

        nan_mask = reduce(np.logical_and, [glob_mask, cpf_mask, npf_mask, vtx_mask])

        return (
            global_arr[nan_mask],
            cpf_arr[nan_mask],
            npf_arr[nan_mask],
            vtx_arr[nan_mask],
            truth_arr[nan_mask],
            process[nan_mask],
        )

    def saveOutput(
        self, output_location, global_arr, cpf_arr, npf_arr, vtx_arr, truth, process
    ):
        np.savez(
            output_location,
            global_features=global_arr,
            cpf_arr=cpf_arr,
            npf_arr=npf_arr,
            vtx_arr=vtx_arr,
            truth=truth,
            process=process,
        )
