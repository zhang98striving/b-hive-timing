import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from BaseTask import MainBaseTask
from dataset.dataset import DatasetConstructorTask
from models.deepjet import DeepJet


class TrainingTask(MainBaseTask):
    def requires(self):
        return DatasetConstructorTask.req(self)

    def output(self):
        return self.local_target("training.txt")

    def run(self):
        string = "hello training!"
        print(string)
        # loading processed numpy files, torch version not tested yet
        with open(f"{self.output_directory}/processed_files.txt", "r") as f:
            if self.fileformat == "numpy":
                dataset = np.concatenate(
                    [np.load(f"{array}") for array in [line.replace("\n", "") for line in f]]
                )
                dataset = torch.tensor(np.expand_dims(dataset, axis=2)).float()
            if self.fileformat == "torch":
                dataset = torch.cat(
                    [
                        torch.load(f"{array}").float()
                        for array in [line.replace("\n", "") for line in f]
                    ]
                )
                dataset = torch.unsqueeze(dataset, 2)
        print(dataset.shape)

        config_dict = np.load("config_dict.npy", allow_pickle=True).item()

        training_data, test_data = random_split(dataset.to(config_dict["device"]), [0.8, 0.2])
        training_data = DataLoader(training_data, batch_size=10000)
        test_data = DataLoader(test_data, batch_size=10000)

        # Model Defintion
        print("Model definition")
        model = DeepJet(config_dict["model"]["feature_edges"]).to(config_dict["device"])

        # Training
        print("Start training")
        train_metrics, test_metrics = perform_training(
            model, training_data, test_data, config_dict, nepochs=3000
        )

        print("Training finished. Saving data...")
        save_dict = {
            "model": model.state_dict(),
            "training_data": training_data,
            "test_data": test_data,
        }
        torch.save(save_dict, "model")
        np.savez(
            "train_metrics", loss=train_metrics[:, 0], acc=train_metrics[:, 1], allow_pickle=True
        )
        np.savez("test_metrics", loss=test_metrics[:, 0], acc=test_metrics[:, 1], allow_pickle=True)
        self.output().dump(string, formatter="text")


class InferenceTask(MainBaseTask):
    def requires(self):
        return TrainingTask.req(self)

    def output(self):
        return self.local_target("inferencetask.txt")

    def run(self):
        model_dict = torch.load("model")
        config_dict = np.load("config_dict.npy", allow_pickle=True).item()
        model = DeepJet(config_dict["model"]["feature_edges"])
        model.load_state_dict(model_dict["model"])
        model.to(device=config_dict["device"])
        model.eval()
        testdata = model_dict["test_data"]
        input = []
        output = []
        for data in testdata:
            # data shape: (batch, input_dim, 1)
            x, y = data[:, :-1, :], data[:, -1, 0]
            with torch.no_grad():
                pred = model(x).cpu().numpy()
                if len(output) == 0:
                    output = pred
                    input = data.cpu().numpy()
                else:
                    output = np.append(output, pred, axis=0)
                    input = np.append(input, data.cpu().numpy(), axis=0)

        np.save("input", input)
        np.save("output", output)
        self.output().dump(f"{input}, {output}", formatter="text")


def train_model(dataloader, model, loss_fn, optimizer, device="cpu"):
    losses = []
    accuracy = 0.0
    model.train()
    for data in dataloader:
        x, y = data[:, :-1, :], data[:, -1, 0]
        pred = model(x)
        loss = loss_fn(pred, y.type(torch.LongTensor).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        accuracy += torch.sum(y == pred.argmax(dim=1))
    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)


def test_model(dataloader, model, loss_fn, device="cpu"):
    losses = []
    accuracy = 0.0
    model.eval()
    for data in dataloader:
        x, y = data[:, :-1, :], data[:, -1, 0]
        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y.type(torch.LongTensor).to(device))
            losses.append(loss.cpu().numpy())
            accuracy += torch.sum(y == pred.argmax(dim=1))
    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)


def perform_training(model, training_data, test_data, config_dict, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    nepochs = kwargs["nepochs"]
    train_metrics = np.zeros((nepochs, 2))
    test_metrics = np.zeros((nepochs, 2))
    for t in range(nepochs):
        print(t, "of", nepochs)
        loss, acc = train_model(training_data, model, loss_fn, optimizer, config_dict["device"])
        train_metrics[t, :] = np.array([loss, acc])
        loss, acc = test_model(test_data, model, loss_fn, config_dict["device"])
        test_metrics[t, :] = np.array([loss, acc])

    return train_metrics, test_metrics


def inference(model, testdata):
    model.eval()
    input = []
    output = []
    for data in testdata:
        # data shape: (batch, input_dim, 1)
        x, y = data[:, :-1, :], data[:, -1, 0]
        with torch.no_grad():
            pred = model(x).cpu().numpy()
            if len(output) == 0:
                output = pred
                input = data.cpu().numpy()
            else:
                output = np.append(output, pred, axis=0)
                input = np.append(input, data.cpu().numpy(), axis=0)

    np.save("input", input)
    np.save("output", output)
    return input, output
