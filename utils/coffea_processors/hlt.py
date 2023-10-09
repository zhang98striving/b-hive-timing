from typing import List

import awkward as ak
import hist
import numpy as np
from coffea import processor

from utils.coffea_processors.base import DataPreprocessing_BaseClass
from utils.dataset.structured_arrays import structured_array_from_tree


class HLTDataPreprocessing(DataPreprocessing_BaseClass):
    def setFeatureNamesAndEdges(self):
        n_cpf = 26
        n_npf = 25
        n_vtx = 5
        feature_edges = []
        feature_names = []
        self.global_features = [
            "jet_pt",
            "jet_eta",
            "nCpfcan",
            "nNpfcan",
            "nsv",
            "npv",
            "TagVarCSV_trackSumJetEtRatio",
            "TagVarCSV_trackSumJetDeltaR",
            "TagVarCSV_vertexCategory",
            "TagVarCSV_trackSip2dValAboveCharm",
            "TagVarCSV_trackSip2dSigAboveCharm",
            "TagVarCSV_trackSip3dValAboveCharm",
            "TagVarCSV_trackSip3dSigAboveCharm",
            "TagVarCSV_jetNSelectedTracks",
            "TagVarCSV_jetNTracksEtaRel",
        ]
        feature_names.append(self.global_features)
        feature_edges.append(len(feature_names))
        self.cpf = [
            "Cpfcan_BtagPf_trackEtaRel",
            "Cpfcan_BtagPf_trackPtRel",
            "Cpfcan_BtagPf_trackPPar",
            "Cpfcan_BtagPf_trackDeltaR",
            "Cpfcan_BtagPf_trackPParRatio",
            "Cpfcan_BtagPf_trackSip2dVal",
            "Cpfcan_BtagPf_trackSip2dSig",
            "Cpfcan_BtagPf_trackSip3dVal",
            "Cpfcan_BtagPf_trackSip3dSig",
            "Cpfcan_BtagPf_trackJetDistVal",
            "Cpfcan_ptrel",
            "Cpfcan_drminsv",
            "Cpfcan_VTX_ass",
            "Cpfcan_puppiw",
            "Cpfcan_chi2",
            "Cpfcan_quality",
        ]
        feature_edges.append(feature_edges[-1] + len(self.cpf) * n_cpf)
        feature_names.extend(self.cpf)
        self.npf = [
            "Npfcan_ptrel",
            "Npfcan_deltaR",
            "Npfcan_isGamma",
            "Npfcan_HadFrac",
            "Npfcan_drminsv",
            "Npfcan_puppiw",
        ]
        feature_edges.append(feature_edges[-1] + len(self.npf) * n_npf)
        feature_names.extend(self.npf)
        self.vtx = [
            "sv_pt",
            "sv_deltaR",
            "sv_mass",
            "sv_ntracks",
            "sv_chi2",
            "sv_normchi2",
            "sv_dxy",
            "sv_dxysig",
            "sv_d3d",
            "sv_d3dsig",
            "sv_costhetasvpv",
            "sv_enratio",
        ]
        feature_edges.append(feature_edges[-1] + len(self.vtx) * n_vtx)
        feature_names.extend(self.vtx)

        feature_names.append("truth")
        feature_names.append("process")

        self.feature_edges = feature_edges
        self.features = feature_names

    def callColumnAccumulator(self, output, events, flag):
        n_cpf = 26
        n_npf = 25
        n_vtx = 5

        # slicing based on p_T and eta
        pt_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) >= min(self.bins_pt),
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) <= max(self.bins_pt),
        )
        eta_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) >= min(self.bins_eta),
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) <= max(self.bins_eta),
        )

        isB = ak.to_numpy(ak.values_astype(ak.flatten(events["isB"], axis=0), np.float32))
        isBB = ak.to_numpy(ak.values_astype(ak.flatten(events["isBB"], axis=0), np.float32))
        isGBB = ak.to_numpy(ak.values_astype(ak.flatten(events["isGBB"], axis=0), np.float32))
        isLeptonicB = ak.to_numpy(
            ak.values_astype(ak.flatten(events["isLeptonicB"], axis=0), np.float32)
        )
        isLeptonicB_C = ak.to_numpy(
            ak.values_astype(ak.flatten(events["isLeptonicB_C"], axis=0), np.float32)
        )
        isC = ak.to_numpy(ak.values_astype(ak.flatten(events["isC"], axis=0), np.float32))
        isCC = ak.to_numpy(ak.values_astype(ak.flatten(events["isCC"], axis=0), np.float32))
        isGCC = ak.to_numpy(ak.values_astype(ak.flatten(events["isGCC"], axis=0), np.float32))
        isUD = ak.to_numpy(ak.values_astype(ak.flatten(events["isUD"], axis=0), np.float32))
        isS = ak.to_numpy(ak.values_astype(ak.flatten(events["isS"], axis=0), np.float32))
        isG = ak.to_numpy(ak.values_astype(ak.flatten(events["isG"], axis=0), np.float32))
        isUndefined = ak.to_numpy(
            ak.values_astype(ak.flatten(events["isUndefined"], axis=0), np.float32)
        )
        isTau = ak.to_numpy(ak.values_astype(ak.flatten(events["isTau"], axis=0), np.float32))
        data_slice = np.array(
            (pt_slice & eta_slice)
            & (
                np.ndarray.astype(isB, np.int32)
                | np.ndarray.astype(isBB, np.int32)
                | np.ndarray.astype(isGBB, np.int32)
                | np.ndarray.astype(isLeptonicB, np.int32)
                | np.ndarray.astype(isLeptonicB_C, np.int32)
                | np.ndarray.astype(isC, np.int32)
                | np.ndarray.astype(isCC, np.int32)
                | np.ndarray.astype(isGCC, np.int32)
                | np.ndarray.astype(isUD, np.int32)
                | np.ndarray.astype(isS, np.int32)
                | np.ndarray.astype(isG, np.int32)
            )
            & np.logical_not(np.ndarray.astype(isUndefined, np.int32))
            & np.logical_not(np.ndarray.astype(isTau, np.int32)),
            dtype=bool,
        )

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
            feature_length=n_cpf,
        )
        npf_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.npf,
            precision=self.precision,
            feature_length=n_npf,
        )
        vtx_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.vtx,
            precision=self.precision,
            feature_length=n_vtx,
        )

        target_class = np.full_like(isB, -999)
        target_class = np.where(isB == 1, 0, target_class)  # b
        target_class = np.where((isBB == 1) | (isGBB == 1), 1, target_class)  # bb
        target_class = np.where(
            (isLeptonicB == 1) | (isLeptonicB_C == 1), 2, target_class
        )  # leptonicb
        target_class = np.where((isC == 1) | (isCC == 1) | (isGCC == 1), 3, target_class)  # c
        target_class = np.where((isUD == 1) | (isS == 1), 4, target_class)  # uds
        target_class = np.where(isG == 1, 5, target_class)  # g

        truth = target_class[data_slice]
        process = np.full_like(target_class[data_slice], flag)

        return global_arr, cpf_arr, npf_arr, vtx_arr, truth, process

    def saveOutput(self, output_location, global_arr, cpf_arr, npf_arr, vtx_arr, truth, process):
        np.savez(
            output_location,
            global_features=global_arr,
            cpf_arr=cpf_arr,
            npf_arr=npf_arr,
            vtx_arr=vtx_arr,
            truth=truth,
            process=process,
        )
