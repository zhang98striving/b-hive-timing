from utils.torch.datasets import DeepJetDataset
from torch.utils.data import DataLoader
from tasks.dataset import DatasetConstructorTask
from utils.models.deepjet import DeepJet
from tasks.base import BaseTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from rich.progress import track
import uproot
import torch.nn as nn
import numpy as np
import luigi
import torch
import math

torch.autograd.detect_anomaly(True)


class TrainingTask(TrainingDependency, DatasetDependency, BaseTask):
    loss_weighting = luigi.BoolParameter(
        True,
        description="Whether to weight the loss or use weighted sampling from the dataset",
    )

    epochs = luigi.IntParameter(default=4)

    def requires(self):
        return DatasetConstructorTask.req(self)

    def output(self):
        return {
            "training_metrics": self.local_target("training_metrics.npz"),
            "validation_metrics": self.local_target("validation_metrics.npz"),
            "model": self.local_target(f"model_{self.epochs-1}.pt"),
            "best_model": self.local_target("best_model.pt"),
        }

    def run(self):
        # Loading config
        config_dict = np.load(self.input()["config_dict"].path, allow_pickle=True).item()

        print("Loading Dataset")
        files = np.array(self.input()["file_list"].load().split("\n")[:-1])
        print(len(files))
        training_mask = ~(np.char.find(files, "train") == -1)
        validation_mask = ~(np.char.find(files, "validation") == -1)

        training_files = files[training_mask]
        validation_files = files[validation_mask]

        histogram_training = np.load(
            self.input()["histogram_training"].path,
            allow_pickle=True,
        )
        training_data = DeepJetDataset(
            training_files,
            "training",
            weighted_sampling=True,  # not self.loss_weighting,
            device=self.device,
            histogram_training=histogram_training,
        )
        validation_data = DeepJetDataset(
            validation_files,
            "validation",
            weighted_sampling=True,  # not self.loss_weighting,
            device=self.device,
            histogram_training=histogram_training,
        )

        batch_size = 1000  # 512
        training_dataloader = DataLoader(
            training_data, batch_size=batch_size, drop_last=True, pin_memory=True
        )  # Pin Memory for faster CPU/GPU memory load
        validation_dataloader = DataLoader(
            validation_data, batch_size=batch_size, drop_last=False, pin_memory=True
        )

        # Model Defintion
        print("Model definition")
        model = DeepJet(config_dict["model"]["feature_edges"]).to(self.device)
        # model = torch.compile(model)
        scaler = torch.cuda.amp.GradScaler()

        # Training
        print("Start training on " + self.device)
        train_metrics, validation_metrics = self.perform_training(
            model,
            training_dataloader,
            validation_dataloader,
            config_dict,
            nepochs=self.epochs,
        )

        print("Training finished. Saving data...")

        np.savez(
            self.output()["training_metrics"].path,
            loss=train_metrics[:, 0],
            acc=train_metrics[:, 1],
            allow_pickle=True,
        )
        np.savez(
            self.output()["validation_metrics"].path,
            loss=validation_metrics[:, 0],
            acc=validation_metrics[:, 1],
            allow_pickle=True,
        )

    def perform_training(self, model, training_data, validation_data, config_dict, **kwargs):
        best_loss_val = math.inf
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        nepochs = kwargs["nepochs"]
        train_metrics = np.zeros((nepochs, 2))
        validation_metrics = np.zeros((nepochs, 2))
        for t in range(nepochs):
            print(t, "of", nepochs)
            loss_train, acc_train = self.train_model(
                training_data,
                model,
                loss_fn,
                optimizer,
                self.device,
            )
            train_metrics[t, :] = np.array([loss_train, acc_train])
            loss_val, acc_val = self.validate_model(validation_data, model, loss_fn, self.device)
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
                "{}/model_{}.pt".format(self.output()["model"].parent.path, t),
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
                    "{}/best_model.pt".format(self.output()["model"].parent.path),
                )

        return train_metrics, validation_metrics

    def train_model(
        self,
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

    def validate_model(self, dataloader, model, loss_fn, device="cpu"):
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


class InferenceTask(TrainingDependency, DatasetDependency, BaseTask):
    def requires(self):
        return {"training": TrainingTask.req(self), "dataset": DatasetConstructorTask.req(self)}

    def output(self):
        return {
            "output_numpy": self.local_target("output.npy"),
            "output_root": self.local_target("output.root"),
        }

    def run(self):
        config_dict = np.load(self.input()["dataset"]["config_dict"].path, allow_pickle=True).item()

        model = DeepJet(config_dict["model"]["feature_edges"]).to(self.device)
        best_model = torch.load(
            self.input()["training"]["best_model"].path,
            map_location=torch.device(self.device),
        )
        model.load_state_dict(best_model["model_state_dict"])

        print("Loading Dataset")
        files = np.array(
            open(self.input()["dataset"]["file_list"].path, "r").read().split("\n")[:-1]
        )
        test_mask = ~(np.char.find(files, "test") == -1)
        test_files = files[test_mask]

        histogram_test = np.load(
            self.input()["dataset"]["histogram_test"].path,
            allow_pickle=True,
        )
        test_data = DeepJetDataset(test_files, "test", histogram_training=histogram_test)
        test_dataloader = DataLoader(test_data, batch_size=1000)

        model.eval()
        kinematics = []
        truth = []
        prediction = []
        output = []
        for x, _, y in track(test_dataloader, "Inference..."):
            x = x.float()

            kinematics.append(x[:, :2, 0])
            truth.append(y)
            with torch.no_grad():
                pred = model(x.to(device=self.device))
                prediction.append(pred)
                if len(output) == 0:
                    output = pred.cpu().numpy()
                else:
                    output = np.append(output, pred.cpu().numpy(), axis=0)

        np.save(self.output()["output_numpy"].path, output)

        prediction = torch.cat(prediction, dim=0).cpu().numpy()
        kinematics = torch.cat(kinematics, dim=0).cpu().numpy()
        truth = torch.cat(truth, dim=0).cpu().numpy().astype(int)
        one_hot_truth = np.zeros((len(truth), np.max(truth) + 1))
        one_hot_truth[np.arange(len(truth)), truth] = 1

        output = np.concatenate((kinematics, prediction, one_hot_truth), axis=1)
        with uproot.recreate(self.output()["output_root"].path) as root_file:
            root_file["tree"] = {
                "Jet_pt": output[:, 0],
                "Jet_eta": output[:, 1],
                "prob_isB": output[:, 2],
                "prob_isBB": output[:, 3],
                "prob_isLeptB": output[:, 4],
                "prob_isC": output[:, 5],
                "prob_isUDS": output[:, 6],
                "prob_isG": output[:, 7],
                "isB": output[:, 8],
                "isBB": output[:, 9],
                "isLeptB": output[:, 10],
                "isC": output[:, 11],
                "isUDS": output[:, 12],
                "isG": output[:, 13],
            }
