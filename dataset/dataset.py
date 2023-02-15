import uproot
import numpy as np
import awkward as ak
import gc

import torch

def pfnano_to_array(rootfile, isMC):
    print('Doing cleaning, isMC = ',isMC)
    
    feature_edges = []

    # Global
    feature_names = ['Jet_pt', 'Jet_eta',
                    'Jet_DeepJet_nCpfcand','Jet_DeepJet_nNpfcand',
                    'Jet_DeepJet_nsv','Jet_DeepJet_npv',
                    'Jet_DeepCSV_trackSumJetEtRatio',
                    'Jet_DeepCSV_trackSumJetDeltaR',
                    'Jet_DeepCSV_vertexCategory',
                    'Jet_DeepCSV_trackSip2dValAboveCharm',
                    'Jet_DeepCSV_trackSip2dSigAboveCharm',
                    'Jet_DeepCSV_trackSip3dValAboveCharm',
                    'Jet_DeepCSV_trackSip3dSigAboveCharm',
                    'Jet_DeepCSV_jetNSelectedTracks',
                    'Jet_DeepCSV_jetNTracksEtaRel'
                    ]
    feature_edges.append(len(feature_names))
    # CPF
    cpf = [[f'Jet_DeepJet_Cpfcan_BtagPf_trackEtaRel_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackPtRel_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackPPar_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackPParRatio_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dVal_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dVal_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dSig_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackJetDistVal_{i}',
            f'Jet_DeepJet_Cpfcan_ptrel_{i}',
            f'Jet_DeepJet_Cpfcan_drminsv_{i}',
            f'Jet_DeepJet_Cpfcan_VTX_ass_{i}',
            f'Jet_DeepJet_Cpfcan_puppiw_{i}',
            f'Jet_DeepJet_Cpfcan_chi2_{i}',
            f'Jet_DeepJet_Cpfcan_quality_{i}'] for i in range(25)]
    feature_names.extend([item for sublist in cpf for item in sublist])
    feature_edges.append(len(feature_names))
    # NPF
    npf = [[f'Jet_DeepJet_Npfcan_ptrel_{i}',
            f'Jet_DeepJet_Npfcan_deltaR_{i}',
            f'Jet_DeepJet_Npfcan_isGamma_{i}',
            f'Jet_DeepJet_Npfcan_HadFrac_{i}',
            f'Jet_DeepJet_Npfcan_drminsv_{i}',
            f'Jet_DeepJet_Npfcan_puppiw_{i}'] for i in range(25)]
    feature_names.extend([item for sublist in npf for item in sublist])
    feature_edges.append(len(feature_names))
    # VTX
    vtx = [[f'Jet_DeepJet_sv_pt_{i}',
            f'Jet_DeepJet_sv_deltaR_{i}',
            f'Jet_DeepJet_sv_mass_{i}',
            f'Jet_DeepJet_sv_ntracks_{i}',
            f'Jet_DeepJet_sv_chi2_{i}',
            f'Jet_DeepJet_sv_normchi2_{i}',
            f'Jet_DeepJet_sv_dxy_{i}',
            f'Jet_DeepJet_sv_dxysig_{i}',
            f'Jet_DeepJet_sv_d3d_{i}',
            f'Jet_DeepJet_sv_d3dsig_{i}',
            f'Jet_DeepJet_sv_costhetasvpv_{i}',
            f'Jet_DeepJet_sv_enratio_{i}'] for i in range(4)]
    feature_names.extend([item for sublist in vtx for item in sublist])
    feature_edges.append(len(feature_names))
    
    number_of_features = len(feature_names)
    
    targets_necessary = True

    if isMC == True and targets_necessary:
        # flavour definition for PFNano based on: https://indico.cern.ch/event/739204/#3-deepjet-overview
        if 'Jet_FlavSplit' in rootfile['Events'].keys():
            feature_names.extend(['Jet_FlavSplit'])
        else:
            feature_names.extend(['Jet_hadronFlavour','Jet_partonFlavour','Jet_nBHadrons'])
     
    print('Events:', rootfile['Events'].num_entries)
    
    # go through a specified number of events, and get the information (awkward-arrays) for the keys specified above
    for data in rootfile['Events'].iterate(feature_names, step_size=rootfile['Events'].num_entries, library='ak'):
        break
    
    # creating an array to store all the columns with their entries per jet, flatten per-event -> per-jet
    # this works ONLY because the number of jets per event will be accessible in the analyzer
    datacolumns = np.zeros((number_of_features+1, len(ak.flatten(data['Jet_pt'], axis=1))))
    #print(len(datacolumns))

    for featureindex in range(number_of_features):
        a = ak.flatten(data[feature_names[featureindex]], axis=1) # flatten along first inside to get jets
        datacolumns[featureindex] = ak.to_numpy(a)

    if isMC == True and targets_necessary:
        if 'Jet_FlavSplit' in rootfile['Events'].keys():
            flavsplit = ak.to_numpy(ak.flatten(data['Jet_FlavSplit'], axis=1))
            # if the list specified below was exhaustive, the -1 would get overwritten all the time
            #target_class = np.full_like(flavsplit, -1)                                                         # initialize
            # but it isn't the case, there are undefined jet flavors, therefore set to something that could be used later
            target_class = np.full_like(flavsplit, 1)                                                         # initialize
            target_class = np.where(flavsplit == 500, 0, target_class)                                                       # b
            target_class = np.where(np.bitwise_or(flavsplit == 510, flavsplit == 511), 1, target_class)                      # bb
            target_class = np.where(np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class)                      # leptonicb
            target_class = np.where(np.bitwise_or(flavsplit == 400, flavsplit == 410, flavsplit == 411), 3, target_class)    # c
            target_class = np.where(np.bitwise_or(flavsplit == 1, flavsplit == 2), 4, target_class)                          # uds
            target_class = np.where(flavsplit == 0, 5, target_class)                                                         # g
            del flavsplit
            gc.collect()
        else: # backup case for samples that don't have fine grained target definition available
            hadronFlav = ak.to_numpy(ak.flatten(data['Jet_hadronFlavour'], axis=1))
            partonFlav = ak.to_numpy(ak.flatten(data['Jet_partonFlavour'], axis=1))
            nBHadrons = ak.to_numpy(ak.flatten(data['Jet_nBHadrons'], axis=1))
            target_class = np.full_like(hadronFlav, 2)                                                         # initialize
            target_class = np.where(np.bitwise_and(hadronFlav == 5, nBHadrons == 1), 0, target_class)                        # b
            target_class = np.where(np.bitwise_and(hadronFlav == 5, nBHadrons > 1.5), 1, target_class)                       # bb
            #target_class = np.where(np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class)                     # leptonicb
            target_class = np.where(hadronFlav == 4, 3, target_class)                                                        # c
            target_class = np.where(np.bitwise_and(hadronFlav != 5, hadronFlav != 4), 4, target_class)                       # uds
            target_class = np.where(np.bitwise_and(hadronFlav != 5, hadronFlav != 4, partonFlav == 21), 5, target_class)     # g
            del hadronFlav
            del partonFlav
            del nBHadrons
            gc.collect()
        datacolumns[number_of_features] = target_class
        
    datavectors = datacolumns.transpose()
    print('Jets:', len(datavectors))    
    # shape of datavectors: number of jets, number of features  +    1
    #                                             inputs           target
    #                                       (both data and MC)   (MC only)
    # Maybe ToDo: wondering whether we need to clean the features like it was done for DeepCSV, the ShallowTagInfos are contained in DeepJet inputs!
    return datavectors, feature_edges

def getDataset(config_dict):
    # Dataset construction
    print("Dataset construction")
    # files = ["QCD_HT100to200.root"] 
    files = ["ttsemileptonic.root"]
    dataset = np.array([])
    for fi in files:
        print(fi)
        file = uproot.open("/hpcwork/rwth1244/PFNano/examples/"+fi)
        output, feature_edges = pfnano_to_array(file, True)
        config_dict["model"]["feature_edges"] = feature_edges
        if len(dataset) == 0:
            dataset = output
        else:
            dataset = np.append(dataset, output, axis=0)
        print("shape:  ", output.shape)
        print("targets:", output[0:30,-1])
    print("dataset shape:", dataset.shape)
    dataset = torch.tensor(np.expand_dims(dataset, axis=2)).float()
    return dataset