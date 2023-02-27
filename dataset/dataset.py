import gc
import hist
import torch
import uproot
import numpy as np
import awkward as ak
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema



def empty_column_accumulator():
    return processor.column_accumulator(np.array([],dtype=np.float64))
def array_accumulator():
    return processor.defaultdict_accumulator(empty_column_accumulator)

class DeepJet_DataPreprocessing(processor.ProcessorABC):
    def __init__(self, features, output_directory):
        self.features     = features
        self._accumulator = processor.dict_accumulator({})
        self.output_dir   = output_directory
        self.lower_pt     = 10
        self.upper_pt     = 2000
        self.lower_eta    = -2.5
        self.upper_eta    = 2.5
        self.bins_pt      = [10,25,30,35,40,45,50,60,75,100,125,150,175,200,250,300,400,500,600,2001]
        self.bins_eta     = [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.6]
        self.b_hist       = (hist.Hist.new.Variable(self.bins_pt, name="pt").Variable(self.bins_eta, name="eta").Int64())
        self.bb_hist      = (hist.Hist.new.Variable(self.bins_pt, name="pt").Variable(self.bins_eta, name="eta").Int64())
        self.lepb_hist    = (hist.Hist.new.Variable(self.bins_pt, name="pt").Variable(self.bins_eta, name="eta").Int64())
        self.c_hist       = (hist.Hist.new.Variable(self.bins_pt, name="pt").Variable(self.bins_eta, name="eta").Int64())
        self.uds_hist     = (hist.Hist.new.Variable(self.bins_pt, name="pt").Variable(self.bins_eta, name="eta").Int64())
        self.g_hist       = (hist.Hist.new.Variable(self.bins_pt, name="pt").Variable(self.bins_eta, name="eta").Int64())
    
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        # extracting strings for saving
        dataset  = events.metadata["dataset"]
        start    = events.metadata["entrystart"]
        stop     = events.metadata["entrystop"]
        filename = events.metadata["filename"].split("/")[-1].strip(".root")
        
        output               = self.accumulator
        output_location_list = []
        
        b_hist    = self.b_hist
        bb_hist   = self.bb_hist
        lepb_hist = self.lepb_hist
        c_hist    = self.c_hist
        uds_hist  = self.uds_hist
        g_hist    = self.g_hist
        
        # slicing based on p_T and eta
        pt_slice   = np.logical_and(ak.to_numpy(ak.flatten(events["Jet"]["pt"], axis=1))>=self.lower_pt, ak.to_numpy(ak.flatten(events["Jet"]["pt"], axis=1))<=self.upper_pt)
        eta_slice  = np.logical_and(ak.to_numpy(ak.flatten(events["Jet"]["eta"], axis=1))>=self.lower_eta, ak.to_numpy(ak.flatten(events["Jet"]["eta"], axis=1))<=self.upper_eta)
        data_slice = np.logical_and(pt_slice, eta_slice)
        
        # storing all features and truth in column accumulator
        for f in self.features[:-1]:
            output[f"Jet_{f}"] = processor.column_accumulator(ak.to_numpy(ak.flatten(events["Jet"][f"{f}"], axis=1))[data_slice])
        flavsplit = ak.to_numpy(ak.flatten(events["Jet"]["FlavSplit"], axis=1))[data_slice]
        target_class = np.full_like(flavsplit, 1)
        target_class = np.where(flavsplit == 500, 0, target_class)                                                    # b
        target_class = np.where(np.bitwise_or(flavsplit == 510, flavsplit == 511), 1, target_class)                   # bb
        target_class = np.where(np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class)                   # leptonicb
        target_class = np.where(np.bitwise_or(flavsplit == 400, flavsplit == 410, flavsplit == 411), 3, target_class) # c
        target_class = np.where(np.bitwise_or(flavsplit == 1, flavsplit == 2), 4, target_class)                       # uds
        target_class = np.where(flavsplit == 0, 5, target_class)                                                      # g
        output[f"Jet_{self.features[-1]}"] = processor.column_accumulator(target_class)
        
        # making the histograms for reweighting
        b_hist.fill(output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 0], output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 0])
        bb_hist.fill(output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 1], output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 1])
        lepb_hist.fill(output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 2], output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 2])
        c_hist.fill(output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 3], output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 3])
        uds_hist.fill(output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 4], output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 4])
        g_hist.fill(output[f"Jet_{self.features[0]}"].value[output[f"Jet_{self.features[-1]}"].value == 5], output[f"Jet_{self.features[1]}"].value[output[f"Jet_{self.features[-1]}"].value == 5])
        
        # saving constructed array in chunks
        output_location = f"{self.output_dir}/{filename}_{start}_{stop}.npy"
        output_location_list.append(output_location)
        np.save(output_location, np.stack([np.concatenate([output[f"Jet_{feature}"].value]) for feature in self.features], axis=1))
        return {"output_location": output_location_list,"b_hist": np.sum([b_hist.view()], axis=0), "bb_hist": np.sum([bb_hist.view()], axis=0), "lepb_hist": np.sum([lepb_hist.view()], axis=0), "c_hist": np.sum([c_hist.view()], axis=0), "uds_hist": np.sum([uds_hist.view()], axis=0), "g_hist": np.sum([g_hist.view()], axis=0)}

    def postprocess(self, accumulator):
        pass
    
    
def getDataset(config_dict):
    print("Dataset construction")
    # defining files to process (TODO: give files as argument to this function)
    sample_dict = {"qcd": ["/hpcwork/rwth1244/PFNano/examples/QCD_HT100to200.root"], "tt": ["/hpcwork/rwth1244/PFNano/examples/ttsemileptonic.root"]}
    
    # defining features to extract
    feature_names = ["pt", "eta", "DeepJet_nCpfcand", "DeepJet_nNpfcand", "DeepJet_nsv", "DeepJet_npv", "DeepCSV_trackSumJetEtRatio", "DeepCSV_trackSumJetDeltaR", "DeepCSV_vertexCategory", "DeepCSV_trackSip2dValAboveCharm", "DeepCSV_trackSip2dSigAboveCharm", "DeepCSV_trackSip3dValAboveCharm", "DeepCSV_trackSip3dSigAboveCharm", "DeepCSV_jetNSelectedTracks", "DeepCSV_jetNTracksEtaRel"]
    cpf = [[f"DeepJet_Cpfcan_BtagPf_trackEtaRel_{i}", f"DeepJet_Cpfcan_BtagPf_trackPtRel_{i}", f"DeepJet_Cpfcan_BtagPf_trackPPar_{i}", f"DeepJet_Cpfcan_BtagPf_trackDeltaR_{i}", f"DeepJet_Cpfcan_BtagPf_trackPParRatio_{i}", f"DeepJet_Cpfcan_BtagPf_trackSip2dVal_{i}", f"DeepJet_Cpfcan_BtagPf_trackSip2dSig_{i}", f"DeepJet_Cpfcan_BtagPf_trackSip3dVal_{i}", f"DeepJet_Cpfcan_BtagPf_trackSip3dSig_{i}", f"DeepJet_Cpfcan_BtagPf_trackJetDistVal_{i}", f"DeepJet_Cpfcan_ptrel_{i}", f"DeepJet_Cpfcan_drminsv_{i}", f"DeepJet_Cpfcan_VTX_ass_{i}", f"DeepJet_Cpfcan_puppiw_{i}", f"DeepJet_Cpfcan_chi2_{i}", f"DeepJet_Cpfcan_quality_{i}"] for i in range(25)]
    feature_names.extend([item for sublist in cpf for item in sublist])
    npf = [[f"DeepJet_Npfcan_ptrel_{i}", f"DeepJet_Npfcan_deltaR_{i}", f"DeepJet_Npfcan_isGamma_{i}", f"DeepJet_Npfcan_HadFrac_{i}", f"DeepJet_Npfcan_drminsv_{i}", f"DeepJet_Npfcan_puppiw_{i}"] for i in range(25)]
    feature_names.extend([item for sublist in npf for item in sublist])
    vtx = [[f"DeepJet_sv_pt_{i}", f"DeepJet_sv_deltaR_{i}", f"DeepJet_sv_mass_{i}", f"DeepJet_sv_ntracks_{i}", f"DeepJet_sv_chi2_{i}", f"DeepJet_sv_normchi2_{i}", f"DeepJet_sv_dxy_{i}", f"DeepJet_sv_dxysig_{i}", f"DeepJet_sv_d3d_{i}", f"DeepJet_sv_d3dsig_{i}", f"DeepJet_sv_costhetasvpv_{i}", f"DeepJet_sv_enratio_{i}"] for i in range(4)]
    feature_names.extend([item for sublist in vtx for item in sublist])
    feature_names.append("truth")
    
    # defining where to save stuff (TODO: give path as argument to this function)
    output_directory = "/hpcwork/rwth1244/PFNano/examples/coffea"
    
    # executing the coffea processor
    futures_run = processor.Runner(executor = processor.FuturesExecutor(compression=None, workers=1), schema=PFNanoAODSchema, chunksize=10000)
    output = futures_run(sample_dict, "Events", processor_instance=DeepJet_DataPreprocessing(feature_names, output_directory))
    
    # saving filelist from coffea
    with open(f"{output_directory}/processed_files.txt", "w") as f:
        for line in output["output_location"]:
            f.write(f"{line}\n")
            
    # saving histograms from coffea
    for key in output.keys():
        if key=="output_location":
            continue
        else:
            output_location = f"{output_directory}/{key}.npy"
            np.save(output_location, output[key])
        
    # loading processed numpy files
    with open(f"{output_directory}/processed_files.txt", "r") as f:
        dataset = np.concatenate([np.load(f"{array}") for array in [line.replace("\n", "") for line in f]])
    
    print("dataset shape:", dataset.shape)
    # converting from numpy array to torch tensor
    dataset = torch.tensor(np.expand_dims(dataset, axis=2)).float()
    
    # hardcoding feature edges (ok until more than DeepJet is supported)
    config_dict["model"]["feature_edges"] = [15, 415, 565, 613]
    return dataset