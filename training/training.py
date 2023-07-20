from torch.utils.data import DataLoader, IterableDataset, random_split
from dataset.dataset import DatasetConstructorTask
from models.deepjet import DeepJet
from BaseTask import MainBaseTask
from rich.progress import track
import torch.nn as nn
import numpy as np
import torch
import math


class TrainingTask(MainBaseTask):
    def requires(self):
        return DatasetConstructorTask.req(self)

    def output(self):
        return self.local_target("test_metrics.npz")

    def run(self):
        # Loading config
        config_dict = np.load(self.output_directory + "/config_dict.npy", allow_pickle=True).item()

        print("Loading Dataset")
        files = np.array(
            open(f"{self.output_directory}/processed_files.txt", "r").read().split("\n")[:-1]
        )
        training_mask = ~(np.char.find(files, "train") == -1)
        validation_mask = ~(np.char.find(files, "validation") == -1)

        training_files = files[training_mask]
        validation_files = files[validation_mask]

        histograms = np.load(f"{self.output_directory}/data_histograms.npy", allow_pickle=True)
        training_data = DeepJetDataset(training_files, histograms)
        validation_data = DeepJetDataset(validation_files, histograms)

        batch_size = 1000
        train_kwargs = {}
        validation_kwargs = {}
        if not self.loss_weighting:
            training_sampler = torch.utils.data.WeightedRandomSampler(
                training_data.get_all_weights(), len(training_data)
            )
            train_kwargs.update({"sampler": training_sampler})

            validation_sampler = torch.utils.data.WeightedRandomSampler(
                validation_data.get_all_weights(), len(validation_data)
            )
            validation_kwargs.update({"sampler": validation_sampler})
        training_dataloader = DataLoader(training_data, batch_size=batch_size, **train_kwargs)
        validation_dataloader = DataLoader(
            validation_data, batch_size=batch_size, **validation_kwargs
        )

        # Model Defintion
        print("Model definition")
        model = DeepJet(config_dict["model"]["feature_edges"]).to(config_dict["device"])

        # Training
        print("Start training")
        train_metrics, test_metrics = self.perform_training(
            model, training_dataloader, validation_dataloader, config_dict, nepochs=1000
        )

        print("Training finished. Saving data...")
        """
        save_dict = {
            "model": model.state_dict(),
            "batch_size": batch_size,
        }
        torch.save(save_dict, f"{self.output_directory}/model")
        """
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

    def perform_training(self, model, training_data, validation_data, config_dict, **kwargs):
        best_loss_val = math.inf
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        nepochs = kwargs["nepochs"]
        train_metrics = np.zeros((nepochs, 2))
        test_metrics = np.zeros((nepochs, 2))
        for t in range(nepochs):
            print(t, "of", nepochs)
            loss_train, acc_train = self.train_model(
                training_data, model, loss_fn, optimizer, config_dict["device"]
            )
            train_metrics[t, :] = np.array([loss_train, acc_train])
            loss_val, acc_val = self.validate_model(
                validation_data, model, loss_fn, config_dict["device"]
            )
            test_metrics[t, :] = np.array([loss_val, acc_val])

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
                f"{self.output_directory}/model_{t}.pt",
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
                    f"{self.output_directory}/best_model.pt",
                )

        return train_metrics, test_metrics

    def train_model(self, dataloader, model, loss_fn, optimizer, device="cpu"):
        losses = []
        accuracy = 0.0
        model.train()
        for data in track(dataloader, "Training..."):
            x, w, y = data[:, :-2, :], data[:, -2, 0], data[:, -1, 0]
            pred = model(x.to(device))
            if self.loss_weighting:
                loss = torch.mean(loss_fn(pred, y.type(torch.LongTensor).to(device)) * w.to(device))
            else:
                loss = torch.mean(loss_fn(pred, y.type(torch.LongTensor).to(device)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
            accuracy += torch.sum(y.to(device) == pred.argmax(dim=1))
        accuracy /= len(dataloader.dataset)
        print("  ", np.array(losses).mean(), float(accuracy))
        return np.array(losses).mean(), float(accuracy)

    def validate_model(self, dataloader, model, loss_fn, device="cpu"):
        losses = []
        accuracy = 0.0
        model.eval()
        for data in track(dataloader, "Validating..."):
            x, w, y = data[:, :-2, :], data[:, -2, 0], data[:, -1, 0]
            with torch.no_grad():
                pred = model(x.to(device))
                if self.loss_weighting:
                    loss = torch.mean(
                        loss_fn(pred, y.type(torch.LongTensor).to(device)) * w.to(device)
                    )
                else:
                    loss = torch.mean(loss_fn(pred, y.type(torch.LongTensor).to(device)))
                losses.append(loss.cpu().numpy())
                accuracy += torch.sum(y.to(device) == pred.argmax(dim=1))
        accuracy /= len(dataloader.dataset)
        print("  ", np.array(losses).mean(), float(accuracy))
        return np.array(losses).mean(), float(accuracy)


class InferenceTask(MainBaseTask):
    def requires(self):
        return TrainingTask.req(self)

    def output(self):
        return self.local_target("output.npy")

    def run(self):
        config_dict = np.load(self.output_directory + "/config_dict.npy", allow_pickle=True).item()

        model = DeepJet(config_dict["model"]["feature_edges"]).to(config_dict["device"])
        best_model = torch.load(
            f"{self.output_directory}/best_model.pt",
            map_location=torch.device(config_dict["device"]),
        )
        model.load_state_dict(best_model["model_state_dict"])

        print("Loading Dataset")
        files = np.array(
            open(f"{self.output_directory}/processed_files.txt", "r").read().split("\n")[:-1]
        )
        test_mask = ~(np.char.find(files, "test") == -1)
        test_files = files[test_mask]
        histograms = np.load(f"{self.output_directory}/data_histograms.npy", allow_pickle=True)
        test_data = DeepJetDataset(test_files, histograms)
        test_dataloader = DataLoader(test_data, batch_size=1000)

        model.eval()
        kinematics = []
        truth = []
        prediction = []
        output = []
        for data in test_dataloader:
            x = data[:, :-2, :]
            kinematics.append(data[:, :2, 0])
            truth.append(data[:, -1, 0])
            with torch.no_grad():
                pred = model(x.to(device=config_dict["device"]))
                prediction.append(pred)
                if len(output) == 0:
                    output = pred.cpu().numpy()
                else:
                    output = np.append(output, pred.cpu().numpy(), axis=0)
                    
        np.save(self.output_directory + "/output.npy", output)
                
        prediction = torch.cat(prediction, dim=0).cpu().numpy()
        kinematics = torch.cat(kinematics, dim=0).cpu().numpy()
        truth = torch.cat(truth, dim=0).cpu().numpy().astype(int)
        one_hot_truth = np.zeros((len(truth), np.max(truth)+1))
        one_hot_truth[np.arange(len(truth)), truth] = 1

        output = np.concatenate((kinematics, prediction, one_hot_truth), axis=1)
        with u.recreate(self.output_directory + "/output.root") as root_file:
            root_file["tree"] = {"Jet_pt": output[:,0], "Jet_eta": output[:,1], "prob_isB": output[:,2], "prob_isBB": output[:,3], "prob_isLeptB": output[:,4], "prob_isC": output[:,5], "prob_isUDS": output[:,6], "prob_isG": output[:,7], "isB": output[:,8], "isBB": output[:,9], "isLeptB": output[:,10], "isC": output[:,11], "isUDS": output[:,12], "isG": output[:,13]}    


class DeepJetDataset(IterableDataset):
    def __init__(self, files, weight_histograms):
        self.files = files
        self.weight_list = self.calculate_weights(weight_histograms)
        self.bins_pt = [
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
        self.bins_eta = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1, 1.5, 2.0, 2.5]
        self.Nedges = [0]
        for file in track(self.files, "Loading dataset..."):
            number_of_samples = np.load(file, allow_pickle=True).shape[0]
            self.Nedges.append(self.Nedges[-1] + number_of_samples)

    def __len__(self):
        return self.Nedges[-1]

    def __getitem__(self, index):
        true_index_in_file = index % self.chunk_size
        loc = index // self.chunk_size
        element = np.load(self.files[loc])[true_index_in_file]
        w = self.get_weight(element)
        element = np.insert(element, -1, w).T
        return torch.tensor(element[:, np.newaxis]).float()

    def __iter__(self):
        for f in self.files:
            for samples in np.load(f):
                w = self.get_weight(samples)
                yield torch.tensor(np.insert(samples, -1, w).T[:, np.newaxis]).float()

    def calculate_weights(self, histograms, classes=6):
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

    def get_weight(self, samples):
        if len(samples.shape) == 1:
            samples = samples.reshape(1, -1)
        pt_coordinate = np.digitize(samples[:, 0], self.bins_pt) - 1
        eta_coordinate = np.digitize(samples[:, 1], self.bins_eta) - 1
        w = np.array(self.weight_list)[
            np.array(samples[:, -1], dtype=int), pt_coordinate, eta_coordinate
        ]
        return w

    def get_all_weights(self):
        weights = []
        for f in track(self.files, "Evaluatings all weights..."):
            samples = np.load(f, allow_pickle=True)
            w = self.get_weight(samples)
            if len(weights) == 0:
                weights = w
            else:
                weights = np.append(weights, w)
        return torch.tensor(weights).float()
