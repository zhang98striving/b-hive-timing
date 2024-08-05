from functools import reduce

import numpy as np
import torch
from numpy.lib import recfunctions
from rich.progress import track
from torch.utils.data import IterableDataset


class PNetDataset(IterableDataset):
    def __init__(
        self,
        files,
        model,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        histogram_training=None,
        max_length=1,
        bins_pt=None,
        bins_eta=None,
        verbose=0,
    ):
        self.verbose = verbose
        self.files = files
        if (bins_pt is None) or (bins_eta is None):
            raise ValueError("You need to specify bins!")
        self.bins_pt = bins_pt
        self.bins_eta = bins_eta
        self.Nedges = [0]
        self.data_type = data_type
        if data_type == "validation":
            self.data_type = "test"
        if histogram_training is not None:
            self.all_number_of_samples = histogram_training.sum()
        else:
            if len(files):
                f = np.load(files[0])
                self.all_number_of_samples = len(f[f.files[0]])
            else:
                self.all_number_of_samples = 0
        self.weighted_sampling = weighted_sampling

        self.device = device
        self.model = model

    def __len__(self):
        return int(self.all_number_of_samples)

    def __getitem__(self, index):
        raise NotImplementedError

    def shuffleFileList(self):
        np.random.shuffle(self.files)

    def __iter__(self):
        # Multi-worker support: each worker gets a separate set of files
        # to iterate over to avoid double iterations
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = self.files
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[
                worker_info.id
            ]

        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")
            with np.load(file) as data:
                # load stuff at once
                global_arrs = data["global_features"]
                truth = data["truth"]
                cpf_arrs = data["cpf_arr"]
                vtx_arrs = data["vtx_arr"]
                weight = data["weight"]
                process = data["process"]
                if self.weighted_sampling:
                    random_number = np.random.rand(len(global_arrs))
                    mask = random_number < weight
                else:
                    mask = np.ones(global_arrs.shape, dtype=np.bool8)

                # truth from all truths to classes
                truths = np.ones(len(truth))
                truth_un = recfunctions.structured_to_unstructured(truth)
                # count up all flavours and assign value
                # this is not nice at all but here we are...

                for index, (name, flavours) in enumerate(self.model.classes.items()):
                    for flav in flavours:
                        truths[truth[flav]] = index

                truths = truths[mask]
                processes = process[mask]
                weights = weight[mask]
                """

                only keep fields that are part of the model

                """
                cpf_points = np.array(
                    [cpf_arrs[point][mask] for point in self.model.cpf_points]
                ).reshape(-1, self.model.n_cpf, 2)
                vtx_points = np.array(
                    [vtx_arrs[point][mask] for point in self.model.vtx_points]
                ).reshape(-1, self.model.n_vtx, 2)

                global_arrs = global_arrs[mask][self.model.global_features]
                cpf_arrs = cpf_arrs[mask][self.model.cpf_candidates]
                vtx_arrs = vtx_arrs[mask][self.model.vtx_features]

                N = len(global_arrs)
                global_arrs = recfunctions.structured_to_unstructured(global_arrs)
                # reshape arrays in (length, candidates, features)
                cpf_arrs = recfunctions.structured_to_unstructured(cpf_arrs).reshape(
                    N, -1, len(cpf_arrs.dtype.names)
                )
                vtx_arrs = recfunctions.structured_to_unstructured(vtx_arrs).reshape(
                    N, -1, len(vtx_arrs.dtype.names)
                )

                cpf_arrs = cpf_arrs[:, : self.model.n_cpf]
                vtx_arrs = vtx_arrs[:, : self.model.n_vtx]
                cpf_points = cpf_points[:, : self.model.n_cpf]
                vtx_points = vtx_points[:, : self.model.n_vtx]
                for (
                    global_a,
                    cpf_a,
                    vtx_a,
                    cpf_point,
                    vtx_point,
                    t,
                    w,
                    p,
                ) in zip(
                    global_arrs,
                    cpf_arrs,
                    vtx_arrs,
                    cpf_points,
                    vtx_points,
                    truths,
                    weights,
                    processes,
                ):
                    yield global_a, cpf_a, vtx_a, cpf_point, vtx_point, t, w, p
            del (
                global_arrs,
                cpf_arrs,
                vtx_arrs,
                cpf_points,
                vtx_points,
                truths,
                weights,
                processes,
                mask,
            )
        return None

    def get_all_weights(self):
        weights = np.empty((self.Nedges[-1]))
        N = 0
        for file in track(
            self.files, "Reading in the weights for the " + self.data_type + " data"
        ):
            with open(file, "rb") as np_file:
                data = np.load(np_file)
            n_elements = int(data.shape[0])
            weights[N : N + n_elements] = data[:, -3]
            N += n_elements
        return weights
