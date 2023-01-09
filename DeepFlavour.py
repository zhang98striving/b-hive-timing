import uproot
import numpy as np
import awkward as ak
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Conv1d, LSTM, Dropout, Linear

from torch.utils.data import DataLoader, random_split

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

# Dataset construction
print("Dataset construction")
files = ["QCD_HT100to200.root"] #["ttsemileptonic.root"]
dataset = np.array([])
for fi in files:
    print(fi)
    file = uproot.open("/hpcwork/rwth1244/PFNano/examples/"+fi)
    targets_necessary = True
    output, feature_edges = pfnano_to_array(file, True)
    if len(dataset) == 0:
        dataset = output
    else:
        dataset = np.append(dataset, output, axis=0)
    print("shape:  ", output.shape)
    print("targets:", output[0:30,-1])
print("dataset shape:", dataset.shape)
dataset = torch.tensor(np.expand_dims(dataset, axis=2)).float()

# Model Defintion
print("Model definition")
class DFModel(nn.Module):
    """
    TODO: weight initialization
    """
    def __init__(self, feature_edges, momentum = 0.6, dropoutRate = 0.1):
        super().__init__()
        self.feature_edges = feature_edges
        self.momentum = momentum
        self.dropoutRate = dropoutRate
        self.bn_gl        = BatchNorm1d(self.feature_edges[0], momentum=self.momentum)
        self.bn_cpf       = BatchNorm1d(self.feature_edges[1]-self.feature_edges[0], momentum=self.momentum)
        self.bn_npf       = BatchNorm1d(self.feature_edges[2]-self.feature_edges[1], momentum=self.momentum)
        self.bn_vtx       = BatchNorm1d(self.feature_edges[3]-self.feature_edges[2], momentum=self.momentum)

        # CPF
        self.conv_cpf1    = Conv1d(in_channels=self.feature_edges[1]-self.feature_edges[0], out_channels=64, kernel_size=1)
        self.bn_cpf1      = BatchNorm1d(64, momentum=self.momentum)
        self.do_cpf1      = Dropout(self.dropoutRate)
        self.conv_cpf2    = Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        self.bn_cpf2      = BatchNorm1d(32, momentum=self.momentum)
        self.do_cpf2      = Dropout(self.dropoutRate)
        self.conv_cpf3    = Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.bn_cpf3      = BatchNorm1d(32, momentum=self.momentum)
        self.do_cpf3      = Dropout(self.dropoutRate)
        self.conv_cpf4    = Conv1d(in_channels=32, out_channels=8, kernel_size=1)

        # NPF
        self.conv_npf1    = Conv1d(in_channels=self.feature_edges[2]-self.feature_edges[1], out_channels=32, kernel_size=1)
        self.bn_npf1      = BatchNorm1d(32, momentum=self.momentum)
        self.do_npf1      = Dropout(self.dropoutRate)
        self.conv_npf2    = Conv1d(in_channels=32, out_channels=16, kernel_size=1)
        self.bn_npf2      = BatchNorm1d(16, momentum=self.momentum)
        self.do_npf2      = Dropout(self.dropoutRate)
        self.conv_npf3    = Conv1d(in_channels=16, out_channels=4, kernel_size=1)

        # VTX
        self.conv_vtx1    = Conv1d(in_channels=self.feature_edges[3]-self.feature_edges[2], out_channels=64, kernel_size=1)
        self.bn_vtx1      = BatchNorm1d(64, momentum=self.momentum)
        self.do_vtx1      = Dropout(self.dropoutRate)
        self.conv_vtx2    = Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        self.bn_vtx2      = BatchNorm1d(32, momentum=self.momentum)
        self.do_vtx2      = Dropout(self.dropoutRate)
        self.conv_vtx3    = Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.bn_vtx3      = BatchNorm1d(32, momentum=self.momentum)
        self.do_vtx3      = Dropout(self.dropoutRate)
        self.conv_vtx4    = Conv1d(in_channels=32, out_channels=8, kernel_size=1)

        # LSTMs
        self.lstm_cpf     = LSTM(input_size=8, hidden_size = 150)
        self.lstm_cpf_bn  = BatchNorm1d(150, momentum=self.momentum)
        self.lstm_cpf_do  = Dropout(self.dropoutRate)

        self.lstm_npf     = LSTM(input_size=4, hidden_size = 50)
        self.lstm_npf_bn  = BatchNorm1d(50, momentum=self.momentum)
        self.lstm_npf_do  = Dropout(self.dropoutRate)

        self.lstm_vtx     = LSTM(input_size=8, hidden_size = 50)
        self.lstm_vtx_bn  = BatchNorm1d(50, momentum=self.momentum)
        self.lstm_vtx_do  = Dropout(self.dropoutRate)

        # Dense layers
        self.d1   = Linear(265, 200)
        self.do_1 = Dropout(self.dropoutRate)
        self.d2   = Linear(200, 100)
        self.do_2 = Dropout(self.dropoutRate)
        self.d3   = Linear(100, 100)
        self.do_3 = Dropout(self.dropoutRate)
        self.d4   = Linear(100, 100)
        self.do_4 = Dropout(self.dropoutRate)
        self.d5   = Linear(100, 100)
        self.do_5 = Dropout(self.dropoutRate)
        self.d6   = Linear(100, 100)
        self.do_6 = Dropout(self.dropoutRate)
        self.d7   = Linear(100, 100)
        self.do_7 = Dropout(self.dropoutRate)
        self.d8 = Linear(100, 100)
        self.do_8 = Dropout(self.dropoutRate)

        # Output layer
        self.pred_layer = Linear(100, 6)
            
    def forward(self, x):
        global_vars = x[...,:self.feature_edges[0],:]
        cpf         = x[...,self.feature_edges[0]:self.feature_edges[1],:]
        npf         = x[...,self.feature_edges[1]:self.feature_edges[2],:]
        vtx         = x[...,self.feature_edges[2]:self.feature_edges[3],:]

        global_vars = self.bn_gl(global_vars)
        cpf = self.bn_cpf(cpf)
        npf = self.bn_npf(npf)
        vtx = self.bn_vtx(vtx)

        cpf = self.do_cpf1(F.relu(self.bn_cpf1(self.conv_cpf1(cpf))))
        cpf = self.do_cpf2(F.relu(self.bn_cpf2(self.conv_cpf2(cpf))))
        cpf = self.do_cpf3(F.relu(self.bn_cpf3(self.conv_cpf3(cpf))))
        cpf = F.relu(self.conv_cpf4(cpf))

        npf = self.do_npf1(F.relu(self.bn_npf1(self.conv_npf1(npf))))
        npf = self.do_npf2(F.relu(self.bn_npf2(self.conv_npf2(npf))))
        npf = F.relu(self.conv_npf3(npf))

        vtx = self.do_vtx1(F.relu(self.bn_vtx1(self.conv_vtx1(vtx))))
        vtx = self.do_vtx2(F.relu(self.bn_vtx2(self.conv_vtx2(vtx))))
        vtx = self.do_vtx3(F.relu(self.bn_vtx3(self.conv_vtx3(vtx))))
        vtx = F.relu(self.conv_vtx4(vtx))

        # LSTM
        # TODO: go_backwards=True in the original implementation
        cpf = self.lstm_cpf_do(self.lstm_cpf_bn(self.lstm_cpf(cpf[...,0])[0]))
        npf = self.lstm_npf_do(self.lstm_npf_bn(self.lstm_npf(npf[...,0])[0]))
        vtx = self.lstm_vtx_do(self.lstm_vtx_bn(self.lstm_vtx(vtx[...,0])[0]))

        x = torch.concat([global_vars[...,0], cpf, npf, vtx], axis=1)
        x = self.do_1(F.relu(self.d1(x)))
        x = self.do_2(F.relu(self.d2(x)))
        x = self.do_3(F.relu(self.d3(x)))
        x = self.do_4(F.relu(self.d4(x)))
        x = self.do_5(F.relu(self.d5(x)))
        x = self.do_6(F.relu(self.d6(x)))
        x = self.do_7(F.relu(self.d7(x)))
        x = self.do_8(F.relu(self.d8(x)))

        x = F.softmax(self.pred_layer(x))
        return x
model = DFModel(feature_edges).to("cuda")

# Training
print("Start training")
training_data, test_data = random_split(dataset.to("cuda"), [0.8, 0.2])
training_data = DataLoader(training_data, batch_size=10000)
test_data     = DataLoader(test_data, batch_size=10000)


def train_model(dataloader, model, loss_fn, optimizer):
    losses = []
    accuracy = 0.0
    model.train()
    for data in dataloader:
        x, y = data[:,:-1,:], data[:,-1,0]
        pred = model(x)
        loss = loss_fn(pred, y.type(torch.LongTensor).to("cuda"))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        accuracy += torch.sum(y == pred.argmax(dim=1))
    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)

def test_model(dataloader, model, loss_fn, optimizer):
    losses = []
    accuracy = 0.0
    model.eval()
    for data in dataloader:
        x, y = data[:,:-1,:], data[:,-1,0]
        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y.type(torch.LongTensor).to("cuda"))
            losses.append(loss.cpu().numpy())
            accuracy += torch.sum(y == pred.argmax(dim=1))
    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
nepochs = 65
train_metrics = np.zeros((nepochs, 2))
test_metrics  = np.zeros((nepochs, 2))
for t in range(nepochs):
    print(t, "of", nepochs)
    loss, acc = train_model(training_data, model, loss_fn, optimizer)
    train_metrics[t] = np.array([loss, acc])
    loss, acc = test_model(test_data, model, loss_fn, optimizer)
    test_metrics[t]  = np.array([loss, acc])
print("Training finished. Saving data...")

torch.save(model.state_dict(), "model.pt")
np.save("train_metrics.npz", train_metrics, allow_pickle=True)
np.save("test_metrics.npz", test_metrics, allow_pickle=True)
print("Done")