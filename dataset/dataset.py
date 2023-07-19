from coffea.nanoevents import BaseSchema, PFNanoAODSchema
from BaseTask import MainBaseTask, config_dict
from rich.progress import track
from coffea import processor
import awkward as ak
import numpy as np
import torch
import hist
import os


class DatasetConstructorTask(MainBaseTask):
    def output(self):
        return self.local_target("processed_files.txt")

    def run(self):
        print("Dataset construction")

        # output_directory = "/home/Matefarkas/phd/service_work/bhive_torch20/output"
        # defining what fileformat to use
        # fileformat = "numpy"  # torch

        # executing the coffea processor
        # defining files to process (TODO: give files as argument to this function)
        # sample_dict = {"qcd": ["/hpcwork/rwth1244/PFNano/examples/QCD_HT100to200.root"], "tt": ["/hpcwork/rwth1244/PFNano/examples/ttsemileptonic.root"]}
        # futures_run = processor.Runner(executor = processor.FuturesExecutor(compression=None, workers=1), schema=PFNanoAODSchema, chunksize=10000)
        # output = futures_run(sample_dict, "Events", processor_instance=DeepJet_DataPreprocessing(output_directory, fileformat, config_dict))

        sample_dict = {
            # "QCD_pt-120To170": [f"/net/scratch/Matefarkas/phd/BTV/2023_01_31/QCD_Pt-120To170_TuneCP5_14TeV-pythia8/BTVtuplesdatasets_phase2_2023_01_31/230131_131848/0000/tuple_{i}.root" for i in range(1, 255)]
            "TT": [
                f"/net/scratch/Matefarkas/phd/BTV/2023_01_27/TT_TuneCP5_14TeV-powheg-pythia8/BTVtuplesdatasets_phase2_2023_01_27/230127_143218/0000/tuple_{i}.root"
                for i in range(1, 749)
            ]
        }
        futures_run = processor.Runner(
            executor=processor.FuturesExecutor(compression=None, workers=1),
            schema=BaseSchema,
            chunksize=10000,
        )
        output = futures_run(
            sample_dict,
            "DeepJetNTupler/DeepJetvars",
            processor_instance=DeepJet_NTupleDataPreprocessing(
                self.output_directory, self.fileformat, config_dict
            ),
        )

        np.save(self.output_directory + "/config_dict", config_dict)

        # saving histograms from coffea
        histograms = []
        file_list = []
        for key in output.keys():
            if key == "output_location":
                continue
            else:
                output_location = f"{self.output_directory}/{key}.npy"
                np.save(output_location, output[key])
                for line in output["output_location"]:
                    file_list.append(f"{line}")
                histograms.append(output[key])
        np.save(self.output_directory + "/data_histograms", np.array(histograms))

        files = np.array(file_list)
        np.random.shuffle(files)
        chunk = []
        chunk_size = 100000
        N_tot = 0
        for f in files:
            N_tot += len(np.load(f))
        N_s = [int(0.6 * N_tot), int(0.2 * N_tot)]
        N_s.append(N_tot - np.array(N_s).sum())
        labels = ["training", "test", "validation"]
        i = 0
        j = 0
        Ns = 0
        output_string = ""
        for file in track(files, "Merging..."):
            data = np.load(file, allow_pickle=True)
            n_samples = data.shape[0]
            index_range = (
                n_samples
                if n_samples + len(chunk) <= self.chunk_size
                else self.chunk_size - len(chunk)
            )
            if n_samples + Ns > N_s[i]:
                index_range = N_s[i] - Ns
            Ns += index_range
            if len(chunk) == 0:
                chunk = data
            else:
                chunk = np.vstack([chunk, data[:index_range]])
            if len(chunk) == chunk_size or Ns == N_s[i]:
                filename = f"{self.output_directory}/{labels[i]}_{j}.npy"
                print("saved", filename)
                np.save(filename, chunk)
                output_string += f"{filename}\n"
                j += 1
                chunk = data[index_range:]
                if Ns == N_s[i]:
                    i += 1
                    Ns = 0
                    j = 0
                if i == 3:
                    break
                Ns += n_samples - index_range
        self.output().dump(f"{output_string}", formatter="text")


def empty_column_accumulator():
    return processor.column_accumulator(np.array([], dtype=np.float64))


def array_accumulator():
    return processor.defaultdict_accumulator(empty_column_accumulator)


class DeepJet_DataPreprocessing(processor.ProcessorABC):
    """
    Extracts features from ROOT files needed for a DeepJet training using a coffea processor. Furthermore, it generates histograms in p_T/eta space for each flavor (b, bb, leptonic b, c, uds, g).
    
    Parameters
    ----------
    self.output_dir : string
                      Defines the directory, where the output will be saved.
    self.format : string
                  Defines the used file format. At the moment only numpy is support.
    self.config_dict : dictionary 
                       The configuration dictionary is used the store and access the used configuration throught the whole framework.
    self._accumulator : array-like
                        Coffea accumulator used to store extracted values in a dictionary. For more infos look at https://github.com/CoffeaTeam/coffea.
    self.lower_pt : float
                    Lower p_T cut applied while extracting features.
    self.upper_pt : float
                    Upper p_T cut applied while extracting features.
    self.lower_eta : float
                     Lower eta cut applied while extracting features.
    self.upper_eta : float
                     Upper eta cut applied while extracting features.
    self.bins_pt : list
                   Binning used for p_T. Due to the behaviour of the hist package, the right most bin had to be modified to ensure compatibility with numpy's binning.
    self.bins_eta : list
                    Binning used for eta. Due to the behaviour of the hist package, the right most bin had to be modified to ensure compatibility with numpy's binning.
    self.b_hist : histogram
                  Initialises the histogram for the flavor b using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.bb_hist : histogram
                   Initialises the histogram for the flavor bb using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.lepb_hist : histogram
                    Initialises the histogram for the flavor leptonic b using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.c_hist : histogram
                Initialises the histogram for the flavor c using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.uds_hist : histogram
                    Initialises the histogram for the flavor uds using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.g_hist : histogram
                  Initialises the histogram for the flavor g using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.setFeatureNamesAndEdges() : 
                                     Function to define the feature names to extract and position in the finale dataset.
    """
    def __init__(self, output_directory, output_fileformat, config_dict):
        self.output_dir = output_directory
        self.format = output_fileformat
        self.config_dict = config_dict
        self._accumulator = processor.dict_accumulator({})
        self.lower_pt = 10
        self.upper_pt = 2000
        self.lower_eta = -2.5
        self.upper_eta = 2.5
        self.bins_pt = [
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
            2000,
            2001,
        ]
        self.bins_eta = [
            -2.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.5,
            1,
            1.5,
            2.0,
            2.5,
            2.6,
        ]

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
        feature_names = [
            "pt",
            "eta",
            "DeepJet_nCpfcand",
            "DeepJet_nNpfcand",
            "DeepJet_nsv",
            "DeepJet_npv",
            "DeepCSV_trackSumJetEtRatio",
            "DeepCSV_trackSumJetDeltaR",
            "DeepCSV_vertexCategory",
            "DeepCSV_trackSip2dValAboveCharm",
            "DeepCSV_trackSip2dSigAboveCharm",
            "DeepCSV_trackSip3dValAboveCharm",
            "DeepCSV_trackSip3dSigAboveCharm",
            "DeepCSV_jetNSelectedTracks",
            "DeepCSV_jetNTracksEtaRel",
        ]
        feature_edges.append(len(feature_names))
        cpf = [
            [
                f"DeepJet_Cpfcan_BtagPf_trackEtaRel_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackPtRel_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackPPar_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackDeltaR_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackPParRatio_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip2dVal_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip2dSig_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip3dVal_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip3dSig_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackJetDistVal_{i}",
                f"DeepJet_Cpfcan_ptrel_{i}",
                f"DeepJet_Cpfcan_drminsv_{i}",
                f"DeepJet_Cpfcan_VTX_ass_{i}",
                f"DeepJet_Cpfcan_puppiw_{i}",
                f"DeepJet_Cpfcan_chi2_{i}",
                f"DeepJet_Cpfcan_quality_{i}",
            ]
            for i in range(25)
        ]
        feature_names.extend([item for sublist in cpf for item in sublist])
        feature_edges.append(len(feature_names))
        npf = [
            [
                f"DeepJet_Npfcan_ptrel_{i}",
                f"DeepJet_Npfcan_deltaR_{i}",
                f"DeepJet_Npfcan_isGamma_{i}",
                f"DeepJet_Npfcan_HadFrac_{i}",
                f"DeepJet_Npfcan_drminsv_{i}",
                f"DeepJet_Npfcan_puppiw_{i}",
            ]
            for i in range(25)
        ]
        feature_names.extend([item for sublist in npf for item in sublist])
        feature_edges.append(len(feature_names))
        vtx = [
            [
                f"DeepJet_sv_pt_{i}",
                f"DeepJet_sv_deltaR_{i}",
                f"DeepJet_sv_mass_{i}",
                f"DeepJet_sv_ntracks_{i}",
                f"DeepJet_sv_chi2_{i}",
                f"DeepJet_sv_normchi2_{i}",
                f"DeepJet_sv_dxy_{i}",
                f"DeepJet_sv_dxysig_{i}",
                f"DeepJet_sv_d3d_{i}",
                f"DeepJet_sv_d3dsig_{i}",
                f"DeepJet_sv_costhetasvpv_{i}",
                f"DeepJet_sv_enratio_{i}",
            ]
            for i in range(4)
        ]
        feature_names.extend([item for sublist in vtx for item in sublist])
        feature_edges.append(len(feature_names))
        feature_names.append("truth")
        self.features = feature_names
        self.feature_edges = feature_edges
        self.config_dict["model"]["feature_edges"] = feature_edges

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        start = events.metadata["entrystart"]
        stop = events.metadata["entrystop"]
        filename = events.metadata["filename"].split("/")[-1].strip(".root")

        output = self.accumulator
        output_location_list = []

        b_hist = self.b_hist
        bb_hist = self.bb_hist
        lepb_hist = self.lepb_hist
        c_hist = self.c_hist
        uds_hist = self.uds_hist
        g_hist = self.g_hist

        pt_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["Jet"]["pt"], axis=1)) >= self.lower_pt,
            ak.to_numpy(ak.flatten(events["Jet"]["pt"], axis=1)) <= self.upper_pt,
        )
        eta_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["Jet"]["eta"], axis=1)) >= self.lower_eta,
            ak.to_numpy(ak.flatten(events["Jet"]["eta"], axis=1)) <= self.upper_eta,
        )
        data_slice = np.logical_and(pt_slice, eta_slice)

        for f in self.features[:-1]:
            output[f"Jet_{f}"] = processor.column_accumulator(
                ak.to_numpy(ak.flatten(events["Jet"][f"{f}"], axis=1))[data_slice]
            )
        flavsplit = ak.to_numpy(ak.flatten(events["Jet"]["FlavSplit"], axis=1))[data_slice]
        target_class = np.full_like(flavsplit, 1)
        target_class = np.where(flavsplit == 500, 0, target_class)  # b
        target_class = np.where(
            np.bitwise_or(flavsplit == 510, flavsplit == 511), 1, target_class
        )  # bb
        target_class = np.where(
            np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class
        )  # leptonicb
        target_class = np.where(
            np.bitwise_or(flavsplit == 400, flavsplit == 410, flavsplit == 411), 3, target_class
        )  # c
        target_class = np.where(
            np.bitwise_or(flavsplit == 1, flavsplit == 2), 4, target_class
        )  # uds
        target_class = np.where(flavsplit == 0, 5, target_class)  # g
        output[f"Jet_{self.features[-1]}"] = processor.column_accumulator(target_class)

        b_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 0],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 0],
        )
        bb_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 1],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 1],
        )
        lepb_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 2],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 2],
        )
        c_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 3],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 3],
        )
        uds_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 4],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 4],
        )
        g_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 5],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 5],
        )

        if self.format == "numpy":
            output_location = f"{self.output_dir}/{filename}_{start}_{stop}.npy"
            output_location_list.append(output_location)
            np.save(
                output_location,
                np.stack(
                    [np.concatenate([output[f"Jet_{feature}"].value]) for feature in self.features],
                    axis=1,
                ),
            )
        if self.format == "torch":
            output_location = f"{self.output_dir}/{filename}_{start}_{stop}.pt"
            output_location_list.append(output_location)
            torch.save(
                torch.from_numpy(
                    np.stack(
                        [
                            np.concatenate([output[f"Jet_{feature}"].value])
                            for feature in self.features
                        ],
                        axis=1,
                    )
                ),
                output_location,
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


class DeepJet_NTupleDataPreprocessing(processor.ProcessorABC):
    def __init__(self, output_directory, output_fileformat, config_dict):
        """
        The NTupleDataProcessor. Inputs expected as follows:
            features: array of strings with the last element standing for "truth"
            output_directory: string
            output_fileformat: string
            config_dict: a dictionary containing the key "model" and within "n_cpf", "n_npf", "n_vtx"
        """
        self.output_dir = output_directory
        self.format = output_fileformat
        self.config_dict = config_dict
        self._accumulator = processor.dict_accumulator({})
        self.lower_pt = 10
        self.upper_pt = 2000
        self.lower_eta = -2.5
        self.upper_eta = 2.5
        self.bins_pt = [
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
            2000,
            2001,
        ]  # one more bin for hist to work as expected
        self.bins_eta = [
            -2.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.5,
            1,
            1.5,
            2.0,
            2.5,
            2.6,
        ]  # one more bin for hist to work as expected

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
        self.feature_edges = feature_edges
        self.features = feature_names
        self.config_dict["model"]["feature_edges"] = feature_edges

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        # extracting strings for saving
        dataset = events.metadata["dataset"]
        start = events.metadata["entrystart"]
        stop = events.metadata["entrystop"]
        filename = events.metadata["filename"].split("/")[-1].split(".")[0]  # strip the .root part

        output = self.accumulator
        output_location_list = []

        b_hist = self.b_hist
        bb_hist = self.bb_hist
        lepb_hist = self.lepb_hist
        c_hist = self.c_hist
        uds_hist = self.uds_hist
        g_hist = self.g_hist

        config_model = self.config_dict["model"]
        n_cpf = config_model["n_cpf"]
        n_npf = config_model["n_npf"]
        n_vtx = config_model["n_vtx"]

        # slicing based on p_T and eta
        pt_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) >= self.lower_pt,
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) <= self.upper_pt,
        )
        eta_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) >= self.lower_eta,
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) <= self.upper_eta,
        )

        isB = ak.to_numpy(ak.flatten(events["isB"], axis=0))
        isBB = ak.to_numpy(ak.flatten(events["isBB"], axis=0))
        isGBB = ak.to_numpy(ak.flatten(events["isGBB"], axis=0))
        isLeptonicB = ak.to_numpy(ak.flatten(events["isLeptonicB"], axis=0))
        isLeptonicB_C = ak.to_numpy(ak.flatten(events["isLeptonicB_C"], axis=0))
        isC = ak.to_numpy(ak.flatten(events["isC"], axis=0))
        isCC = ak.to_numpy(ak.flatten(events["isCC"], axis=0))
        isGCC = ak.to_numpy(ak.flatten(events["isGCC"], axis=0))
        isUD = ak.to_numpy(ak.flatten(events["isUD"], axis=0))
        isS = ak.to_numpy(ak.flatten(events["isS"], axis=0))
        isG = ak.to_numpy(ak.flatten(events["isG"], axis=0))
        isUndefined = ak.to_numpy(ak.flatten(events["isUndefined"], axis=0))
        isTau = ak.to_numpy(ak.flatten(events["isTau"], axis=0))
        data_slice = np.array(
            (pt_slice & eta_slice)
            & (
                isB
                | isBB
                | isGBB
                | isLeptonicB
                | isLeptonicB_C
                | isC
                | isCC
                | isGCC
                | isUD
                | isS
                | isG
            )
            & np.logical_not(isUndefined)
            & np.logical_not(isTau),
            dtype=bool,
        )

        # storing all features and truth in column accumulator
        # Global variables
        for f in self.features[: self.feature_edges[0]]:
            arr = events[f"{f}"][data_slice]
            output[f"Jet_{f}"] = processor.column_accumulator(ak.to_numpy(arr))
        # Charged particles
        for i in range(n_cpf):
            for f in [fi for fi in self.features if "Cpfcan_" in fi]:
                arr = events[f"{f}"][data_slice]
                arr = ak.to_numpy(ak.fill_none(ak.pad_none(arr, n_cpf)[:, :n_cpf], 0))
                output[f"Jet_{f}_{i}"] = processor.column_accumulator(arr[:, i])
        # Neutral particles
        for i in range(n_npf):
            for f in [fi for fi in self.features if "Npfcan_" in fi]:
                arr = events[f"{f}"][data_slice]
                arr = ak.to_numpy(ak.fill_none(ak.pad_none(arr, n_npf)[:, :n_npf], 0))
                output[f"Jet_{f}_{i}"] = processor.column_accumulator(arr[:, i])
        # Secondary vertices
        for i in range(n_vtx):
            for f in [fi for fi in self.features if "sv_" in fi]:
                arr = events[f"{f}"][data_slice]
                arr = ak.to_numpy(ak.fill_none(ak.pad_none(arr, n_vtx)[:, :n_vtx], 0))
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
        output[f"Jet_{self.features[-1]}"] = processor.column_accumulator(target_class[data_slice])

        # making the histograms for reweighting
        b_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 0],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 0],
        )
        bb_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 1],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 1],
        )
        lepb_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 2],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 2],
        )
        c_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 3],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 3],
        )
        uds_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 4],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 4],
        )
        g_hist.fill(
            output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 5],
            output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 5],
        )

        # saving constructed array in chunks, torch version not tested yet
        if self.format == "numpy":
            output_location = f"{self.output_dir}/{dataset}_{filename}_{start}_{stop}.npy"
            output_location_list.append(output_location)
            np.save(
                output_location,
                np.stack(
                    [np.concatenate([output[f"{feature}"].value]) for feature in output.keys()],
                    axis=1,
                ),
            )
        if self.format == "torch":
            output_location = f"{self.output_dir}/{filename}_{start}_{stop}.pt"
            output_location_list.append(output_location)
            torch.save(
                torch.from_numpy(
                    np.stack(
                        [np.concatenate([output[f"{feature}"].value]) for feature in output.keys()],
                        axis=1,
                    )
                ),
                output_location,
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
