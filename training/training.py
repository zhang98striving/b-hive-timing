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
        return self.local_target("test_metrics.npz")

    def run(self):
        # Loading config
        config_dict = np.load(self.output_directory + "/config_dict.npy", allow_pickle=True).item()
        training_data, test_data = getDataset(self.output_directory, self.fileformat, config_dict)

        batch_size = 1000

        training_sampler = torch.utils.data.WeightedRandomSampler(
            training_data.dataset[training_data.indices, -2, 0], len(training_data.indices)
        )
        training_dataloader = DataLoader(
            training_data, batch_size=batch_size, sampler=training_sampler
        )

        test_sampler = torch.utils.data.WeightedRandomSampler(
            test_data.dataset[test_data.indices, -2, 0], len(test_data.indices)
        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

        # Model Defintion
        print("Model definition")
        model = DeepJet(config_dict["model"]["feature_edges"]).to(config_dict["device"])

        # Training
        print("Start training")
        train_metrics, test_metrics = perform_training(
            model, training_dataloader, test_dataloader, config_dict, nepochs=1
        )

        print("Training finished. Saving data...")
        save_dict = {
            "model": model.state_dict(),
            "batch_size": batch_size,
        }
        torch.save(save_dict, f"{self.output_directory}/model")
        np.savez(
            self.output_directory + "/train_metrics",
            loss=train_metrics[:, 0],
            acc=train_metrics[:, 1],
            allow_pickle=True,
        )
        np.savez(
            self.output_directory + "/test_metrics",
            loss=test_metrics[:, 0],
            acc=test_metrics[:, 1],
            allow_pickle=True,
        )
        # self.output().dump(string, formatter="text")


class InferenceTask(MainBaseTask):
    def requires(self):
        return TrainingTask.req(self)

    def output(self):
        return self.local_target("output.npy")

    def run(self):
        config_dict = np.load(self.output_directory + "/config_dict.npy", allow_pickle=True).item()
        model_dict = torch.load(f"{self.output_directory}/model")
        _, test_data = getDataset(self.output_directory, self.fileformat, config_dict)
        test_sampler = torch.utils.data.WeightedRandomSampler(
            test_data.dataset[test_data.indices, -2, 0], len(test_data.indices)
        )
        test_data = DataLoader(test_data, batch_size=model_dict["batch_size"], sampler=test_sampler)
        model = DeepJet(config_dict["model"]["feature_edges"])
        model.load_state_dict(model_dict["model"])
        model.to(device=config_dict["device"])
        model.eval()
        input = []
        output = []
        for data in test_data:
            x, y = data[:, :-2, :], data[:, -1, 0]
            with torch.no_grad():
                pred = model(x.to(device=config_dict["device"])).cpu().numpy()
                if len(output) == 0:
                    output = pred
                    input = data.cpu().numpy()
                else:
                    output = np.append(output, pred, axis=0)
                    input = np.append(input, data.cpu().numpy(), axis=0)

        np.save(self.output_directory + "/input", input)
        np.save(self.output_directory + "/output", output)


def getDataset(output_directory, fileformat, config_dict):
    # loading processed numpy files
    # FIXME: torch version has no support for weights yet!
    histograms = np.load(f"{output_directory}/data_histograms.npy", allow_pickle=True)
    with open(f"{output_directory}/processed_files.txt", "r") as f:
        if fileformat == "numpy":
            dataset = np.concatenate(
                [np.load(f"{array}") for array in [line.replace("\n", "") for line in f]]
            )
            dataset = assign_weights_to_jets(dataset, histograms)
            dataset = (
                torch.tensor(np.expand_dims(dataset, axis=2)).float().to(config_dict["device"])
            )
        if fileformat == "torch":
            dataset = torch.cat(
                [torch.load(f"{array}").float() for array in [line.replace("\n", "") for line in f]]
            )
            dataset = torch.unsqueeze(dataset, 2).to(config_dict["device"])
    training_data, test_data = random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(1)
    )
    return training_data, test_data


def train_model(dataloader, model, loss_fn, optimizer, device="cpu"):
    losses = []
    accuracy = 0.0
    model.train()
    for data in dataloader:
        x, y = data[:, :-2, :], data[:, -1, 0]
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
        x, y = data[:, :-2, :], data[:, -1, 0]
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
        x, y = data[:, :-2, :], data[:, -1, 0]
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


def calculate_weights(histograms, classes=6):
    reference_histogram = histograms[0]
    reference_histogram = reference_histogram / np.max(reference_histogram)

    weights_list = []
    for c in range(classes):
        other_histogram = histograms[c]
        other_histogram = other_histogram / np.max(other_histogram)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(other_histogram > 0, reference_histogram / other_histogram, -10)
        weights = weights / np.max(weights)

        weights[weights < 0] = 1
        weights[weights == np.nan] = 1

        weights = weights / np.mean(weights)

        weights_list.append(weights)
    return weights_list


def assign_weights_to_jets(samples, histograms):
    bins_pt = [
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
    ]
    bins_eta = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1, 1.5, 2.0, 2.5]

    b_weight, bb_weight, lepb_weight, c_weight, uds_weight, g_weight = calculate_weights(histograms)

    pt_coordinate = np.digitize(samples[:, 0], bins_pt) - 1
    eta_coordinate = np.digitize(samples[:, 1], bins_eta) - 1

    is_b = np.isin(samples[:, -1], 0)
    is_bb = np.isin(samples[:, -1], 1)
    is_lepb = np.isin(samples[:, -1], 2)
    is_c = np.isin(samples[:, -1], 3)
    is_uds = np.isin(samples[:, -1], 4)
    is_g = np.isin(samples[:, -1], 5)

    assigned_weights = np.empty(np.shape(samples)[0])

    assigned_weights = np.where(is_b, b_weight[pt_coordinate, eta_coordinate], assigned_weights)
    assigned_weights = np.where(is_bb, bb_weight[pt_coordinate, eta_coordinate], assigned_weights)
    assigned_weights = np.where(
        is_lepb, lepb_weight[pt_coordinate, eta_coordinate], assigned_weights
    )
    assigned_weights = np.where(is_c, c_weight[pt_coordinate, eta_coordinate], assigned_weights)
    assigned_weights = np.where(is_uds, uds_weight[pt_coordinate, eta_coordinate], assigned_weights)
    assigned_weights = np.where(is_g, g_weight[pt_coordinate, eta_coordinate], assigned_weights)

    samples = np.append(samples, np.reshape(assigned_weights, (-1, 1)), axis=1)
    samples[:, [-1, -2]] = samples[:, [-2, -1]]
    return samples
