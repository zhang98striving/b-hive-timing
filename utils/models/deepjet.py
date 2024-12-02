import numpy as np
import torch
import torch.nn as nn

from utils.torch import LZ4Dataset
from utils.plotting.termplot import terminal_roc
from utils.models.base_model import Classifier_base
from utils.plotting.termplot import terminal_roc
from utils.models.helpers import (
    DenseClassifier,
    MoDDenseClassifier,
    InputProcess,
    MoDInputProcess,
)
from scipy.special import softmax

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.special import softmax

torch.multiprocessing.set_sharing_strategy("file_system")

class DeepJet(Classifier_base):

    def __init__(self,
                 config,
                 cpf_conv = [64, 32, 32, 8],
                 npf_conv = [32, 16, 4],
                 vtx_conv = [64, 32, 32, 8],
                 global_dim = 15,
                 n_layers_lstm = 1,
                 lstm_dim = [150, 50, 50],
                 dense_clas_dim = [200, 100, 100, 100, 100, 100, 100, 100, 100],
                 **kwargs
        ):
        
        super(DeepJet, self).__init__(**kwargs)

        global_dim = len(config['global_features'])
        cpf_conv = [len(config['cpf_candidates'])] + cpf_conv
        npf_conv = [len(config['npf_candidates'])] + npf_conv
        vtx_conv = [len(config['vtx_features'])] + vtx_conv
        
        self.InputProcess = InputProcess(cpf_conv, npf_conv, vtx_conv)

        dense_clas_dim_full = [sum(lstm_dim) + global_dim] + dense_clas_dim
        self.DenseClassifier = DenseClassifier(dense_clas_dim_full)

        self.global_bn = torch.nn.BatchNorm1d(global_dim, eps=0.001, momentum=0.6)
        self.cpf_lstm = torch.nn.LSTM(
            input_size=cpf_conv[-1], hidden_size=lstm_dim[0], num_layers=n_layers_lstm, batch_first=True
        )
        self.npf_lstm = torch.nn.LSTM(
            input_size=npf_conv[-1], hidden_size=lstm_dim[1],  num_layers=n_layers_lstm, batch_first=True
        )
        self.vtx_lstm = torch.nn.LSTM(
            input_size=vtx_conv[-1], hidden_size=lstm_dim[2],  num_layers=n_layers_lstm, batch_first=True
        )

        self.cpf_bn = torch.nn.BatchNorm1d(lstm_dim[0], eps=0.001, momentum=0.6)
        self.npf_bn = torch.nn.BatchNorm1d(lstm_dim[1], eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(lstm_dim[2], eps=0.001, momentum=0.6)

        self.cpf_dropout = nn.Dropout(0.1)
        self.npf_dropout = nn.Dropout(0.1)
        self.vtx_dropout = nn.Dropout(0.1)

        self.Linear = nn.Linear(100, len(self.classes))
        
    def forward(self, inpt):
         
        global_features, cpf_features, npf_features, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        global_features = global_features[:, :self.input_dims[0][1]]
        cpf_features = cpf_features[:, :, :self.input_dims[1][1]]
        npf_features = npf_features[:, :, :self.input_dims[2][1]]
        vtx_features = vtx_features[:, :, :self.input_dims[3][1]]
        
        global_features = self.global_bn(global_features)
        
        cpf, npf, vtx = self.InputProcess(cpf_features, npf_features, vtx_features)
        
        cpf = self.cpf_lstm(torch.flip(cpf, dims=[1]))[0][:, -1]
        cpf = self.cpf_dropout(self.cpf_bn(cpf))
            
        npf = self.npf_lstm(torch.flip(npf, dims=[1]))[0][:, -1]
        npf = self.npf_dropout(self.npf_bn(npf))

        vtx = self.vtx_lstm(torch.flip(vtx, dims=[1]))[0][:, -1]
        vtx = self.vtx_dropout(self.vtx_bn(vtx))

        fts = torch.cat((global_features, cpf, npf, vtx), dim=1)
        fts = self.DenseClassifier(fts)

        output = self.Linear(fts)

        return output

"""
class DeepJetHLT(DeepJet):

    cpf_candidates = [
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

    npf_candidates = [
        "Npfcan_ptrel",
        "Npfcan_deltaR",
        "Npfcan_isGamma",
        "Npfcan_HadFrac",
        "Npfcan_drminsv",
        "Npfcan_puppiw",
    ]

    vtx_features = [
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

    global_features = [
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
"""

class MoDJet(DeepJet):
    n_cpf = 25
    n_npf = 25
    n_vtx = 5

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isU", "isD", "isS"],
        "g": ["isG"],
    }

    global_features = [
        "jet_px",
        "jet_py",
        "jet_pz",
        "jet_energy",
        "jet_mass",
        "n_Cpfcand",
        "n_Npfcand",
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

    cpf_candidates = [
        "Cpfcan_px",
        "Cpfcan_py",
        "Cpfcan_pz",
        "Cpfcan_e",
        "Cpfcan_mass",
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

    npf_candidates = [
        "Npfcan_px",
        "Npfcan_py",
        "Npfcan_pz",
        "Npfcan_e",
        "Npfcan_mass",
        "Npfcan_ptrel",
        "Npfcan_deltaR",
        "Npfcan_isGamma",
        "Npfcan_HadFrac",
        "Npfcan_drminsv",
        "Npfcan_puppiw",
    ]

    vtx_features = [
        "sv_px",
        "sv_py",
        "sv_pz",
        "sv_e",
        "sv_mass",
        "sv_deltaR",
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

    feature_keys = [global_features, cpf_candidates, npf_candidates, vtx_features]

    def __init__(self, feature_edges=[18, 543, 818, 893], **kwargs):
        super(MoDJet, self).__init__(**kwargs)

        self.feature_edges = np.array(feature_edges)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.InputProcess = MoDInputProcess()
        self.DenseClassifier = MoDDenseClassifier()

        self.global_bn = torch.nn.BatchNorm1d(18, eps=0.001, momentum=0.6)
        self.cpf_lstm = torch.nn.LSTM(
            input_size=8, hidden_size=150, num_layers=1, batch_first=True
        )
        self.npf_lstm = torch.nn.LSTM(
            input_size=4, hidden_size=50, num_layers=1, batch_first=True
        )
        self.vtx_lstm = torch.nn.LSTM(
            input_size=8, hidden_size=50, num_layers=1, batch_first=True
        )

        self.cpf_bn = torch.nn.BatchNorm1d(150, eps=0.001, momentum=0.6)
        self.npf_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)

        self.cpf_dropout = nn.Dropout(0.1)
        self.npf_dropout = nn.Dropout(0.1)
        self.vtx_dropout = nn.Dropout(0.1)

        self.Linear = nn.Linear(100, len(self.classes))

        self.glob_integers = torch.tensor([5, 6, 7, 8, 11, 16, 17])
        self.cpf_integers = torch.tensor([17, 18, 19, 20])
        self.npf_integers = torch.tensor([7])
        self.vtx_integers = torch.tensor([6])
        self.integers = [
            self.glob_integers,
            self.cpf_integers,
            self.npf_integers,
            self.vtx_integers,
        ]
        self.glob_defaults = torch.tensor([0])
        self.cpf_defaults = torch.tensor([0])
        self.npf_defaults = torch.tensor([0])
        self.vtx_defaults = torch.tensor([0])
        self.defaults = [
            self.glob_defaults,
            self.cpf_defaults,
            self.npf_defaults,
            self.vtx_defaults,
        ]