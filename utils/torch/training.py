import math

import numpy as np
import torch
import torch.nn as nn
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.special import softmax

from utils.plotting.termplot import terminal_roc


def perform_training(
    model,
    training_data,
    validation_data,
    directory,
    device,
    learning_rate=0.001,
    **kwargs,
):
    best_loss_val = math.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    nepochs = kwargs["nepochs"]
    train_metrics = np.zeros((nepochs, 2))
    validation_metrics = np.zeros((nepochs, 2))
    print("Initial ROC")
    _, _ = validate_model(validation_data, model, loss_fn, device)
    for t in range(nepochs):
        print("Epoch", t + 1, "of", nepochs)
        loss_train, acc_train = train_model(
            training_data,
            model,
            loss_fn,
            optimizer,
            device,
        )
        train_metrics[t, :] = np.array([loss_train, acc_train])

        loss_val, acc_val = validate_model(validation_data, model, loss_fn, device)
        validation_metrics[t, :] = np.array([loss_val, acc_val])

        torch.save(
            {
                "epoch": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_train": loss_train,
                "acc_train": acc_train,
                "loss_val": loss_val,
                "acc_val": acc_val,
            },
            "{}/model_{}.pt".format(directory, t),
        )

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss_train,
                    "acc_train": acc_train,
                    "loss_val": loss_val,
                    "acc_val": acc_val,
                },
                "{}/best_model.pt".format(directory),
            )

    return train_metrics, validation_metrics


def train_model(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device="cpu",
):
    losses = []
    accuracy = 0.0
    model.train()

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
        for (
            global_features,
            cpf_features,
            npf_features,
            vtx_features,
            truth,
            weight,
            process,
        ) in dataloader:
            pred = model(
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
                "?"
                if dataloader.nits_expected == len(dataloader)
                else f"~{dataloader.nits_expected}",
            )
        progress.update(task, completed=dataloader.nits_expected)
    dataloader.nits_expected = N // dataloader.batch_size
    accuracy /= N
    print("  ", f"Average loss: {np.array(losses).mean():.4f}")
    print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
    return np.array(losses).mean(), float(accuracy)


def validate_model(dataloader, model, loss_fn, device="cpu"):
    losses = []
    accuracy = 0.0
    model.eval()

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
                pred = model(
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
                    (pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
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
                "?"
                if dataloader.nits_expected == len(dataloader)
                else f"~{dataloader.nits_expected}",
            )
        progress.update(task, completed=dataloader.nits_expected)
    dataloader.nits_expected = N // dataloader.batch_size
    accuracy /= N
    terminal_roc(predictions, truths, title="Validation ROC")

    print("  ", f"Average loss: {np.array(losses).mean():.4f}")
    print("  ", f"Average accuracy: {float(accuracy):.4f}")
    return np.array(losses).mean(), float(accuracy)
