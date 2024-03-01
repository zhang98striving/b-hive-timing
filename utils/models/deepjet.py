import numpy as np
import torch
import torch.nn as nn

from utils.models.abstract_base_models import Classifier
from utils.torch import DeepJetDataset
from utils.plotting.termplot import terminal_roc
from utils.models.helpers import DenseClassifier, InputProcess
from scipy.special import softmax

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class DeepJet(Classifier, nn.Module):
    n_cpf = 25
    n_npf = 25
    n_vtx = 5
    datasetClass = DeepJetDataset
    optimizerClass = torch.optim.Adam

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isUD", "isS"],
        "g": ["isG"],
    }

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

    def __init__(self, feature_edges=[15, 415, 565, 613], **kwargs):
        super(DeepJet, self).__init__(**kwargs)

        self.feature_edges = np.array(feature_edges)
        self.InputProcess = InputProcess()
        self.DenseClassifier = DenseClassifier()

        self.global_bn = torch.nn.BatchNorm1d(15, eps=0.001, momentum=0.6)
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

    def forward(self, global_features, cpf_features, npf_features, vtx_features):
        global_features = self.global_bn(global_features)
        # cpf = cpf_features.reshape(cpf_features.shape[0], 25, 16)
        # npf = npf_features.reshape(npf_features.shape[0], 25, 6)
        # vtx = vtx_features.reshape(vtx_features.shape[0], 4, 12)

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

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        optimizer=None,
        device=None,
        nepochs=0,
        resume_epochs=0,
        **kwargs,
    ):
        best_loss_val = np.inf
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss_train = []
        acc_train = []
        loss_val = []
        acc_val = []

        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        # print("Initial ROC")

        # _, _ = self.validate_model(validation_data, loss_fn, device)
        for t in range(resume_epochs, nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            loss_trainining, acc_training = self.update(
                training_data,
                loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )
            loss_train += loss_trainining 
            acc_train.append(acc_training)

            loss_validation, acc_validation = self.validate_model(validation_data, loss_fn, device)
            loss_val += loss_validation
            acc_val.append(acc_validation)


            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss_train,
                    "acc_train": acc_train,
                    "loss_val": loss_val,
                    "acc_val": acc_val,
                },
                "{}/model_{}.pt".format(directory, t),
            )

            if np.mean(loss_validation) < best_loss_val:
                best_loss_val = np.mean(loss_validation)
                torch.save(
                    {
                        "epoch": t,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_train": loss_train,
                        "acc_train": acc_train,
                        "loss_val": loss_val,
                        "acc_val": acc_val,
                    },
                    "{}/best_model.pt".format(directory),
                )

        return loss_train, loss_val, acc_train, acc_val,

    def predict_model(self, dataloader, device):
        self.eval()
        kinematics = []
        truths = []
        processes = []
        predictions = []
        for (
            global_features,
            cpf_features,
            npf_features,
            vtx_features,
            truth,
            weight,
            process,
        ) in dataloader:
            # append jet_pt and jet_eta
            # this should be done differenlty in the future... avoid array slicing with magic numbers!
            with torch.no_grad():
                pred = self(
                    *[
                        feature.float().to(device)
                        for feature in [
                            global_features,
                            cpf_features,
                            npf_features,
                            vtx_features,
                        ]
                    ]
                )
            kinematics.append(global_features[..., :2].cpu().numpy())
            truths.append(truth.cpu().numpy().astype(int))
            processes.append(process.cpu().numpy())
            predictions.append(pred.cpu().numpy().astype(int))

        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths)
        processes = np.concatenate(processes)
        return predictions, truths, kinematics, processes

    def update(
        self,
        dataloader,
        loss_fn,
        optimizer,
        scaler=None,
        device="cpu",
        verbose=True,
    ):
        losses = []
        accuracy = 0.0
        self.train()

        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Training...", total=dataloader.nits_expected)
            print("entering traing loop")
            for (
                global_features,
                cpf_features,
                npf_features,
                vtx_features,
                truth,
                weight,
                process,
            ) in dataloader:
                with torch.autocast(
                    device_type=device, enabled=True if device == "cuda" else False
                ):  # We select either cuda float16 mixed precision or cpu float32 as LSTMs does not accept bfloat16
                    pred = self.forward(
                        *[
                            feature.float().to(device)
                            for feature in [
                                global_features,
                                cpf_features,
                                npf_features,
                                vtx_features,
                            ]
                        ]
                    )
                    loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()

                if scaler != None:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())
                accuracy += (
                    (pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
                )
                N += len(pred)
                progress.update(
                    task, advance=1, description=f"Training...   | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
        return losses, accuracy

    def validate_model(self, dataloader, loss_fn, device="cpu", verbose=True):
        losses = []
        accuracy = 0.0
        self.eval()

        predictions = np.empty((0, 6))
        truths = np.empty((0))
        processes = np.empty((0))

        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Validation...", total=dataloader.nits_expected)
            for (
                global_features,
                cpf_features,
                npf_features,
                vtx_features,
                truth,
                weight,
                process,
            ) in dataloader:
                with torch.no_grad():
                    pred = self.forward(
                        *[
                            feature.float().to(device)
                            for feature in [
                                global_features,
                                cpf_features,
                                npf_features,
                                vtx_features,
                            ]
                        ]
                    )
                    loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()
                    losses.append(loss.item())

                    accuracy += (
                        (pred.argmax(1) == truth.to(device))
                        .type(torch.float)
                        .sum()
                        .item()
                    )
                    predictions = np.append(predictions, pred.to("cpu").numpy(), axis=0)
                    truths = np.append(truths, truth.to("cpu").numpy(), axis=0)
                    processes = np.append(processes, process.to("cpu").numpy(), axis=0)
                N += global_features.size(dim=0)
                progress.update(
                    task, advance=1, description=f"Validation... | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Validation loss: {np.array(losses).mean():.4f}")
        print("  ", f"Validation accuracy: {float(accuracy):.4f}")

        if verbose:
            print("Printing terminal ROC")
            terminal_roc(predictions, truths, title="Validation ROC")

        return losses, float(accuracy)

    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_jets = (truth == 0) | (truth == 1) | (truth == 2)
        c_jets = truth == 3
        l_jets = (truth == 4) | (truth == 5)
        summed_jets = b_jets + c_jets + l_jets

        b_pred = predictions[:, :3].sum(axis=1)
        c_pred = predictions[:, 3]
        l_pred = predictions[:, -2:].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)
        bvsall = np.where(
            (b_pred + l_pred + c_pred) > 0, (b_pred) / (b_pred + l_pred + c_pred), -1
        )

        b_veto = (truth != 0) & (truth != 1) & (truth != 2) & (summed_jets != 0)
        c_veto = (truth != 3) & (summed_jets != 0)
        l_veto = (truth != 4) & (truth != 5) & (summed_jets != 0)
        no_veto = np.ones(b_veto.shape, dtype=np.bool)

        labels = ["bvsl", "bvsc", "cvsb", "cvsl", "bvsall"]
        discs = [bvsl, bvsc, cvsb, cvsl, bvsall]
        vetos = [c_veto, l_veto, l_veto, b_veto, no_veto]
        truths = [b_jets, b_jets, c_jets, c_jets, b_jets]
        xlabels = [
            "b-identification",
            "b-identification",
            "c-identification",
            "c-identification",
            "b-identification",
        ]
        ylabels = ["light mis-id.", "c mis-id", "b mis-id.", "light mis-id.", "mis-id."]

        return discs, truths, vetos, labels, xlabels, ylabels


class DeepJetHLT(DeepJet):
    n_cpf = 25
    n_npf = 25
    n_vtx = 5
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
