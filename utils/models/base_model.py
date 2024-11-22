import math
import torch
import torch.nn as nn
import os
import numpy as np
from functools import partial

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.plotting.termplot import terminal_roc
from utils.config.config_loader import ConfigLoader
from scipy.special import softmax


class CustomTimeElapsedColumn(TimeElapsedColumn):
    def __init__(self):
        super().__init__()
        self.elapsed_time = 0

    def render(self, task):
        self.elapsed_time = task.elapsed
        return super().render(task)
        
class Classifier_base(nn.Module):

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isUD", "isS"],
        "g": ["isG"],
    }

    # integer positions and default values still have to be checked
    glob_integers = torch.tensor([2, 3, 4, 5, 8, 13, 14])
    cpf_integers = torch.tensor([12, 13, 14, 15])
    npf_integers = torch.tensor([2])
    vtx_integers = torch.tensor([3])
    integers = [
        glob_integers,
        cpf_integers,
        npf_integers,
        vtx_integers,
    ]
    glob_defaults = torch.tensor([0])
    cpf_defaults = torch.tensor([0])
    npf_defaults = torch.tensor([0])
    vtx_defaults = torch.tensor([0])
    defaults = [
        glob_defaults,
        cpf_defaults,
        npf_defaults,
        vtx_defaults,
    ]

    
    def create_feature_lengths(self, config):
        config = ConfigLoader.load_config(config)
        # Constructions of input shape from config.yaml file
        self.input_dims = [
            (1,                          len(config['global_features'])), 
            (config['n_cpf_candidates'], len(config['cpf_candidates'])), 
            (config['n_npf_candidates'], len(config['npf_candidates'])), 
            (config['n_vtx_candidates'], len(config['vtx_features']))
        ]
        
        feature_edges = []
        v = 0
        for dim in self.input_dims:
            v += dim[0]*dim[1]
            feature_edges.append(v)
    
        feature_edges = torch.Tensor(feature_edges).int()    
        feature_lengths = feature_edges[1:] - feature_edges[:-1]
        self.feature_lengths = torch.cat((feature_edges[:1], feature_lengths))
        

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        attack=None,
        optimizer=None,
        scheduler=None,
        batch_lr=False,
        device=None,
        nepochs=0,
        best_loss_val = np.inf,
        resume_epochs=0,
        train_metrics=None,
        validation_metrics=None,
        terminal_plot=False,
        **kwargs,
    ):
        
        if self.use_torch_compile:
            self.compile_step = torch.compile(self.step, mode='max-autotune')
            
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        scaler = torch.amp.GradScaler(device)
        
        if os.path.isfile(f'{directory}/train_time.npy') and os.path.isfile(f'{directory}/val_time.npy'):
            train_time = np.load(f'{directory}/train_time.npy')
            val_time   = np.load(f'{directory}/val_time.npy')
        else:
            train_time, val_time = np.zeros(nepochs-resume_epochs), np.zeros(nepochs-resume_epochs)
            
        for t in range(resume_epochs, nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            loss_training, acc_training, train_time[t] = self.update(
                training_data,
                loss_fn,
                optimizer,
                scheduler=scheduler,
                batch_lr=batch_lr,
                attack=attack,
                scaler=scaler,
                device=device,
            )
            train_metrics["loss"].append(loss_training)
            train_metrics["acc"].append(acc_training)

            if (not batch_lr) and (scheduler is not None):
                scheduler.step()

            loss_validation, acc_validation, val_time[t] = self.validate_model(validation_data, loss_fn, device,terminal_plot=terminal_plot)
            
            validation_metrics["loss"].append(loss_validation)
            validation_metrics["acc"].append(acc_validation)

            # Save the model state and other details
            checkpoint = {
                "epoch": t,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_train": loss_training,
                "acc_train": acc_training,
                "loss_val": loss_validation,
                "acc_val": acc_validation,
            }
            
            # Save the current model
            torch.save(checkpoint, f"{directory}/model_{t}.pt")
            
            # Save the best model if current validation loss is lower
            if loss_validation < best_loss_val:
                best_loss_val = loss_validation
                torch.save(checkpoint, f"{directory}/best_model.pt")

            # Save time taken for training and validation
            np.save(f'{directory}/train_time.npy', train_time)
            np.save(f'{directory}/val_time.npy', val_time)

            np.savez(
                f'{directory}/training_metrics',
                loss=train_metrics["loss"],
                acc=train_metrics["acc"],
                allow_pickle=True,
            )
            np.savez(
                f'{directory}/validation_metrics',
                loss=validation_metrics["loss"],
                acc=validation_metrics["acc"],
                allow_pickle=True,
            )
        
        return train_metrics, validation_metrics

    def predict_model(
        self, 
        dataloader, 
        device, 
        attack=None
    ):
        self.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        kinematics = []
        truths = []
        processes = []
        predictions = []

        elapsed_column = CustomTimeElapsedColumn()
        
        with Progress(
            TextColumn("{task.description}"),
            elapsed_column,
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 1
            task = progress.add_task("Inference...", total=dataloader.nits_expected)
            
            for (x, truth, w, process) in dataloader:

                x = x.float().to(device, non_blocking=True)
                truth = truth.float().to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                torch.backends.cudnn.enabled = False
                inpt, _ = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)
                torch.backends.cudnn.enabled = True
                
                with torch.no_grad():
                    pred = self(inpt)
                    loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()

                kinematics.append(inpt[0][..., :2].cpu().numpy())
                truths.append(truth.cpu().numpy().astype(int))
                processes.append(process.cpu().numpy())
                predictions.append(pred.cpu().numpy())
                
                N += len(pred)
                progress.update(
                    task, advance=1, description=f"Inference...   | Loss: {loss:.2f}"
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

        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths)
        processes = np.concatenate(processes)
        
        return predictions, truths, kinematics, processes, elapsed_column.elapsed_time
    
    #@torch.compile(mode='max-autotune')
    def step(self, x, truth, loss_fn, attack=None, device="cpu", mixed_precision=True):

        inpt, truth = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)

        if mixed_precision:
            with torch.autocast(device):
                pred = self.forward(inpt)
                loss = loss_fn(pred, truth).mean()
        else:
            pred = self.forward(inpt)
            loss = loss_fn(pred, truth).mean()
            
        return pred, loss
    
    def update(
        self,
        dataloader,
        loss_fn,
        optimizer,
        scheduler=None,
        batch_lr=False,
        attack=None,
        scaler=None,
        device="cpu",
        verbose=True,
    ):
        losses = []
        accuracy = 0.0
        self.train()

        elapsed_column = CustomTimeElapsedColumn()

        with Progress(
            TextColumn("{task.description}"),
            elapsed_column,
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 1
            task = progress.add_task("Training...", total=dataloader.nits_expected)
            print("entering traing loop")
            for (x, truth, w, p) in dataloader:

                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                if self.use_torch_compile:
                    pred, loss = self.compile_step(x, truth, loss_fn, attack=attack, device=device, mixed_precision=self.mixed_precision)
                else:
                    pred, loss = self.step(x, truth, loss_fn, attack=attack, device=device, mixed_precision=self.mixed_precision)

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

                if batch_lr and (scheduler is not None):
                    scheduler.step()
      
                losses.append(loss.item())
                accuracy += (
                    (pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
                )

                N += len(pred)
                curr_lr = optimizer.param_groups[0]['lr']
                progress.update(
                    task, advance=1, description=f"Training...   | Loss: {loss:.4f}, lr: {curr_lr:.5f}"
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

        return np.array(losses).mean(), float(accuracy), elapsed_column.elapsed_time

    def validate_model(self, dataloader, loss_fn, device="cpu", verbose=True, terminal_plot=False):
        losses = []
        accuracy = 0.0
        self.eval()

        predictions = np.empty((0, 6))
        truths = np.empty((0))
        processes = np.empty((0))

        elapsed_column = CustomTimeElapsedColumn()

        with Progress(
            TextColumn("{task.description}"),
            elapsed_column,
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 1
            task = progress.add_task("Validation...", total=dataloader.nits_expected)
            for (x, truth, w, process) in dataloader:
                
                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                with torch.no_grad():
                    if self.use_torch_compile:
                        pred, loss = self.compile_step(x, truth, loss_fn, attack=None, device=device, mixed_precision=self.mixed_precision)
                    else:
                        pred, loss = self.step(x, truth, loss_fn, attack=None, device=device, mixed_precision=self.mixed_precision)
                        
                    losses.append(loss.item())

                    accuracy += (
                        (pred.argmax(1) == truth.to(device))
                        .type(torch.float)
                        .sum()
                        .item()
                    )
                    if(terminal_plot):
                        predictions = np.append(predictions, pred.to("cpu").numpy(), axis=0)
                        truths = np.append(truths, truth.to("cpu").numpy(), axis=0)
                        processes = np.append(processes, process.to("cpu").numpy(), axis=0)
                    
                N += len(pred)
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
        print("  ", f"Validation accuracy: {float(100*accuracy):.4f}")

        if verbose and terminal_plot:
            print("Printing terminal ROC")
            terminal_roc(predictions, truths, title="Validation ROC")

        return np.array(losses).mean(), float(accuracy), elapsed_column.elapsed_time

    def get_inpt(self, x, truth=None, loss_fn=None, attack=None, device='cpu'):

        glob, cpf, npf, vtx = x.split(self.feature_lengths.tolist(), dim=1)
        
        glob = glob.reshape(glob.shape[0], -1)
        cpf = cpf.reshape(cpf.shape[0], self.input_dims[1][0], -1)
        npf = npf.reshape(npf.shape[0], self.input_dims[2][0], -1)
        vtx = vtx.reshape(vtx.shape[0], self.input_dims[3][0], -1)

        if attack is not None:
            (
                glob,
                cpf,
                npf,
                vtx,
                truth,
            ) = attack(
                [
                    feature.float().to(device)
                    for feature in [
                        glob,
                        cpf,
                        npf,
                        vtx,
                        ]
                    ],
                truth.type(torch.LongTensor).to(device),
                loss_fn,
                self,
            ) 
        
        return (glob.detach(), cpf.detach(), npf.detach(), vtx.detach()), truth

    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_jets = (truth == 0) | (truth == 1) | (truth == 2)
        c_jets = truth == 3
        uds_jets = truth == 4
        g_jets = truth == 5
        l_jets = uds_jets | g_jets
        summed_jets = b_jets + c_jets + l_jets

        b_pred = predictions[:, :3].sum(axis=1)
        c_pred = predictions[:, 3]
        uds_pred = predictions[:, 4]
        g_pred = predictions[:, 5]
        l_pred = predictions[:, -2:].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)
        bvsall = np.where(
            (b_pred + l_pred + c_pred) > 0, (b_pred) / (b_pred + l_pred + c_pred), -1
        )
        uds_vs_g = np.where((uds_pred + g_pred) > 0, (uds_pred) / (uds_pred + g_pred), -1)

        b_veto = (truth != 0) & (truth != 1) & (truth != 2) & (summed_jets != 0)
        c_veto = (truth != 3) & (summed_jets != 0)
        bc_veto = (truth != 0) & (truth != 1) & (truth != 2) & (truth != 3) & (summed_jets != 0)
        l_veto = (truth != 4) & (truth != 5) & (summed_jets != 0)
        no_veto = np.ones(b_veto.shape, dtype=np.bool)

        labels = ["bvsl", "bvsc", "cvsb", "cvsl", "bvsall", "uds_vs_g"]
        discs = [bvsl, bvsc, cvsb, cvsl, bvsall, uds_vs_g]
        vetos = [c_veto, l_veto, l_veto, b_veto, no_veto, bc_veto]
        truths = [b_jets, b_jets, c_jets, c_jets, b_jets, uds_jets]
        xlabels = [
            "b-identification",
            "b-identification",
            "c-identification",
            "c-identification",
            "b-identification",
            "uds-identification",
        ]
        ylabels = ["light mis-id.", "c mis-id", "b mis-id.", "light mis-id.", "mis-id.", "gluons mis-id."]

        return discs, truths, vetos, labels, xlabels, ylabels
