import math
import torch
import numpy as np
import torch.nn as nn

from rich.progress import track


def perform_training(model, training_data, validation_data, directory, device, **kwargs):
    best_loss_val = math.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    nepochs = kwargs["nepochs"]
    train_metrics = np.zeros((nepochs, 2))
    validation_metrics = np.zeros((nepochs, 2))
    for t in range(nepochs):
        print(t, "of", nepochs)
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
    it = 0
    for x, w, y in track(dataloader, "Training..."):
        pred = model(x.float().to(device))
        loss = loss_fn(pred, y.type(torch.LongTensor).to(device)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracy += torch.sum(y.to(device) == pred.argmax(dim=1))
        it += 1

    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)


def validate_model(dataloader, model, loss_fn, device="cpu"):
    losses = []
    accuracy = 0.0
    model.eval()
    for x, w, y in track(dataloader, "Validating..."):
        with torch.no_grad():
            pred = model(x.float().to(device))
            loss = loss_fn(pred, y.type(torch.LongTensor).to(device)).mean()
            losses.append(loss.item())

            accuracy += torch.sum(y.to(device) == pred.argmax(dim=1))
    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)
