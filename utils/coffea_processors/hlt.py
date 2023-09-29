import awkward as ak
import hist
import numpy as np

from coffea import processor
from typing import List

from utils.coffea_processors.base import DataPreprocessing_BaseClass

class HLTDataPreprocessing(DataPreprocessing_BaseClass):
    def setFeatureNamesAndEdges(self):
        n_cpf = self.config_dict["model"]["n_cpf"]
        n_npf = self.config_dict["model"]["n_npf"]
        n_vtx = self.config_dict["model"]["n_vtx"]
        feature_edges = []
        feature_names = [
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
        feature_edges.append(len(feature_names))
        cpf = [
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
        feature_edges.append(feature_edges[-1] + len(cpf) * n_cpf)
        feature_names.extend(cpf)
        npf = [
            "Npfcan_ptrel",
            "Npfcan_deltaR",
            "Npfcan_isGamma",
            "Npfcan_HadFrac",
            "Npfcan_drminsv",
            "Npfcan_puppiw",
        ]
        feature_edges.append(feature_edges[-1] + len(npf) * n_npf)
        feature_names.extend(npf)
        vtx = [
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
        feature_edges.append(feature_edges[-1] + len(vtx) * n_vtx)
        feature_names.extend(vtx)

        feature_names.append("truth")
        feature_names.append("process")

        self.feature_edges = feature_edges
        self.features = feature_names
        self.config_dict["model"]["feature_edges"] = feature_edges

    def callColumnAccumulator(self, output, events, flag):
        config_model = self.config_dict["model"]
        n_cpf = config_model["n_cpf"]
        n_npf = config_model["n_npf"]
        n_vtx = config_model["n_vtx"]

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

        # storing all features and truth in column accumulator
        # Global variables
        for f in self.features[: self.feature_edges[0]]:
            arr = events[f"{f}"][data_slice]
            output[f"Jet_{f}"] = processor.column_accumulator(
                ak.to_numpy(ak.values_astype(arr, np.float32))
            )
        # Charged particles
        for i in range(n_cpf):
            for f in [fi for fi in self.features if "Cpfcan_" in fi]:
                arr = events[f"{f}"][data_slice]
                arr = ak.to_numpy(
                    ak.values_astype(
                        ak.fill_none(ak.pad_none(arr, n_cpf)[:, :n_cpf], 0), np.float32
                    )
                )
                output[f"Jet_{f}_{i}"] = processor.column_accumulator(arr[:, i])
        # Neutral particles
        for i in range(n_npf):
            for f in [fi for fi in self.features if "Npfcan_" in fi]:
                arr = events[f"{f}"][data_slice]
                arr = ak.to_numpy(
                    ak.values_astype(
                        ak.fill_none(ak.pad_none(arr, n_npf)[:, :n_npf], 0), np.float32
                    )
                )
                output[f"Jet_{f}_{i}"] = processor.column_accumulator(arr[:, i])
        # Secondary vertices
        for i in range(n_vtx):
            for f in [fi for fi in self.features if "sv_" in fi]:
                arr = events[f"{f}"][data_slice]
                arr = ak.to_numpy(
                    ak.values_astype(
                        ak.fill_none(ak.pad_none(arr, n_vtx)[:, :n_vtx], 0), np.float32
                    )
                )
                output[f"Jet_{f}_{i}"] = processor.column_accumulator(arr[:, i])

        target_class = np.full_like(isB, -999)
        target_class = np.where(isB == 1, 0, target_class)  # b
        target_class = np.where((isBB == 1) | (isGBB == 1), 1, target_class)  # bb
        target_class = np.where(
            (isLeptonicB == 1) | (isLeptonicB_C == 1), 2, target_class
        )  # leptonicb
        target_class = np.where((isC == 1) | (isCC == 1) | (isGCC == 1), 3, target_class)  # c
        target_class = np.where((isUD == 1) | (isS == 1), 4, target_class)  # uds
        target_class = np.where(isG == 1, 5, target_class)  # g

        output["Jet_truth"] = processor.column_accumulator(target_class[data_slice])
        output["Jet_process"] = processor.column_accumulator(
            np.full_like(target_class[data_slice], flag)
        )

        return output

    def saveOutput(self, output_location, output):
        arr = np.stack(
            [np.concatenate([output[f"{feature}"].value]) for feature in output.keys()],
            axis=1,
        )
        # clean NaNs
        arr = arr[~np.any(np.isnan(arr), axis=-1)]
        np.save(output_location, arr)