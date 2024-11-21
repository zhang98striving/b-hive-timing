import numpy as np
import torch
from functools import reduce
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
                if self.weighted_sampling:
                    random_number = np.random.rand(len(data["global_features"]))
                    mask = random_number < data["weight"]
                else:
                    mask = np.ones(data["global_features"].shape, dtype=np.bool8)

                # truth from all truths to classes
                truths = np.ones(len(data["truth"]))
                truth_un = recfunctions.structured_to_unstructured(data["truth"])
                # count up all flavours and assign value
                # this is not nice at all but here we are...

                for index, (name, flavours) in enumerate(self.model.classes.items()):
                    for flav in flavours:
                        truths[data["truth"][flav]] = index

                truths = truths[mask]
                processes = data["process"][mask]
                weights = data["weight"][mask]
                """

                only keep fields that are part of the model

                """
                global_arrs = recfunctions.drop_fields(
                    data["global_features"][mask],
                    [
                        f
                        for f in data["global_features"].dtype.names
                        if f not in self.model.global_features
                    ],
                )
                cpf_arrs = recfunctions.drop_fields(
                    data["cpf_arr"][mask],
                    [
                        f
                        for f in data["cpf_arr"].dtype.names
                        if not f in self.model.cpf_candidates
                    ],
                )
                vtx_arrs = recfunctions.drop_fields(
                    data["vtx_arr"][mask],
                    [
                        f
                        for f in data["vtx_arr"].dtype.names
                        if not f in self.model.vtx_features
                    ],
                )

                N = len(global_arrs)
                global_arrs = recfunctions.structured_to_unstructured(global_arrs)
                # reshape arrays in (length, candidates, features)
                cpf_arrs = recfunctions.structured_to_unstructured(cpf_arrs).reshape(
                    N, -1, len(cpf_arrs.dtype.names)
                )
                vtx_arrs = recfunctions.structured_to_unstructured(vtx_arrs).reshape(
                    N, -1, len(vtx_arrs.dtype.names)
                )

                cpf_points = np.array(
                    [data["cpf_arr"][point][mask] for point in self.model.cpf_points]
                ).reshape(-1, self.model.n_cpf, 2)
                vtx_points = np.array(
                    [data["vtx_arr"][point][mask] for point in self.model.vtx_points]
                ).reshape(-1, self.model.n_vtx, 2)

                for (
                    global_arr,
                    cpf_arr,
                    vtx_arr,
                    cpf_point,
                    vtx_point,
                    truth,
                    weight,
                    process,
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
                    # trim down to number of candidates
                    cpf_arr = cpf_arr[: self.model.n_cpf]
                    vtx_arr = vtx_arr[: self.model.n_vtx]
                    cpf_points = cpf_points[: self.model.n_cpf]
                    vtx_points = vtx_points[: self.model.n_vtx]
                    yield global_arr, cpf_arr, vtx_arr, cpf_point, vtx_point, truth, weight, process
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
