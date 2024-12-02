import awkward as ak
import numpy as np
from functools import reduce

from utils.coffea_processors.base import DataPreprocessing_BaseClass
from utils.dataset.structured_arrays import (
    structured_array_from_tree,
    structured_array_from_tree_truth_from_dict,
)


class LZ4Processing(DataPreprocessing_BaseClass):
    def callColumnAccumulator(self, output, events, flag, **kwargs):
        # slicing based on p_T and eta
        pt_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events[self.pt_key], axis=0)) >= min(self.bins_pt),
            ak.to_numpy(ak.flatten(events[self.pt_key], axis=0)) < max(self.bins_pt),
        )
        eta_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events[self.eta_key], axis=0)) >= min(self.bins_eta),
            ak.to_numpy(ak.flatten(events[self.eta_key], axis=0)) < max(self.bins_eta),
        )

        if isinstance(self.truths, dict):
            truth_arr = structured_array_from_tree_truth_from_dict(
                events=events,
                truth_dict=self.truths,
                precision=np.bool8,
                feature_length=1,
            )
        else:
            truth_arr = structured_array_from_tree(
                events=events,
                keys=self.truths,
                precision=np.bool8,
                feature_length=1,
            )

        data_slice = np.array(
            (pt_slice & eta_slice)
            & reduce(
                np.logical_or,
                [
                    truth_arr[truth]
                    for truth in (
                        self.truths.keys()
                        if isinstance(self.truths, dict)
                        else self.truths
                    )
                ],
            ),
            dtype=bool,
        )
        truth_arr = truth_arr[data_slice]

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

        # create an array with the process value
        process = np.ones(len(global_arr)) * flag

        glob_mask = reduce(
            np.logical_and,
            [~np.any(np.isnan(global_arr[key])) for key in global_arr.dtype.names],
        )
        if cpf_arr.dtype.names:
            cpf_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(cpf_arr[key]), axis=1)
                    for key in cpf_arr.dtype.names
                ],
            )
        else:
            cpf_mask = np.ones(len(global_arr))
        if npf_arr.dtype.names:
            npf_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(npf_arr[key]), axis=1)
                    for key in npf_arr.dtype.names
                ],
            )
        else:
            npf_mask = np.ones(len(global_arr))
        if vtx_arr.dtype.names:
            vtx_mask = reduce(
                np.logical_and,
                [~np.any(np.isnan(vtx_arr[key]), axis=1) for key in vtx_arr.dtype.names],
            )
        else:
            vtx_mask = np.ones(len(global_arr))

        nan_mask = reduce(np.logical_and, [glob_mask, cpf_mask, npf_mask, vtx_mask])

        return (
            global_arr[nan_mask],
            cpf_arr[nan_mask],
            npf_arr[nan_mask],
            vtx_arr[nan_mask],
            truth_arr[nan_mask],
            process[nan_mask],
        )

    def process_array(self, arr, n_jet, n_Cand=None):
        """
        Safely process an array by checking if it's valid and non-empty.
        Returns a reshaped version or a placeholder array if the array is empty.
        """
        if len(arr) == 0 or not len(arr.dtype.names):
            # Return a placeholder array with the correct shape
            if n_Cand == None:
                return np.zeros((n_jet, 0), dtype=np.float32)
            else:
                return np.zeros((n_jet, 0, n_Cand), dtype=np.float32) 
        return arr.view((arr.dtype[0], len(arr.dtype.names))).astype(np.float32)
        
    def saveOutput(
        self, output_location, global_arr, cpf_arr, npf_arr, vtx_arr, truth, process
    ):
        n_jet = truth.shape[0]
        
        # Process individual arrays, replacing missing arrays with placeholders
        global_part = self.process_array(global_arr, n_jet).reshape(n_jet, -1)
        cpf_part = np.swapaxes(self.process_array(cpf_arr, n_jet, n_Cand=self.n_cpf), 1, 2).reshape(n_jet, -1)
        npf_part = np.swapaxes(self.process_array(npf_arr, n_jet, n_Cand=self.n_npf), 1, 2).reshape(n_jet, -1)
        vtx_part = np.swapaxes(self.process_array(vtx_arr, n_jet, n_Cand=self.n_vtx), 1, 2).reshape(n_jet, -1)
        process_part = process.astype(np.float32).reshape(n_jet, -1)
        truth_part = truth.view((truth.dtype[0], len(truth.dtype.names))).astype(np.float32).reshape(n_jet, -1)
        
        arr = np.concatenate([global_part, cpf_part, npf_part, vtx_part, process_part, truth_part], axis=1)
        
        arr = arr[~np.any(np.isnan(arr), axis=-1)]
        arr = arr[~np.any(np.isinf(arr), axis=-1)]

        np.save(
            output_location[:-4],
            arr,
        )
