import numpy as np
import torch
from rich.progress import track
from torch.utils.data import IterableDataset
from numpy.lib import recfunctions


class DeepJetDataset(IterableDataset):
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
        self.all_number_of_samples = histogram_training.sum()
        self.weighted_sampling = weighted_sampling

        self.device = device
        self.model = model

    def __len__(self):
        print("Retunring length: {}", int(self.all_number_of_samples))
        return int(self.all_number_of_samples)

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        # Multi-worker support: each worker gets a separate set of files
        # to iterate over to avoid double iterations
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = self.files
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[worker_info.id]

        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")
            with np.load(file) as data:
                if self.weighted_sampling:
                    random_number = np.random.rand(len(data["global_features"]))
                    mask = random_number < data["weight"]
                else:
                    mask = np.ones(data["global_features"].shape, dtype=np.bool8)

                truths = data["truth"][mask]
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
                    [f for f in data["cpf_arr"].dtype.names if not f in self.model.cpf_candidates],
                )
                npf_arrs = recfunctions.drop_fields(
                    data["npf_arr"][mask],
                    [f for f in data["npf_arr"].dtype.names if not f in self.model.npf_candidates],
                )
                vtx_arrs = recfunctions.drop_fields(
                    data["vtx_arr"][mask],
                    [f for f in data["vtx_arr"].dtype.names if not f in self.model.vtx_features],
                )
                i = 0
                for global_arr, cpf_arr, npf_arr, vtx_arr, truth, weight, process in zip(
                    global_arrs, cpf_arrs, npf_arrs, vtx_arrs, truths, weights, processes
                ):
                    i += 1
                    # this should yield flat arrays with the dedicated features
                    global_arr = recfunctions.structured_to_unstructured(global_arr)
                    cpf_arr = recfunctions.structured_to_unstructured(cpf_arr)
                    npf_arr = recfunctions.structured_to_unstructured(npf_arr)
                    vtx_arr = recfunctions.structured_to_unstructured(vtx_arr)
                    yield global_arr, cpf_arr, npf_arr, vtx_arr, truth, weight, process
        return None

    def get_all_weights(self):
        weights = np.empty((self.Nedges[-1]))
        N = 0
        for file in track(self.files, "Reading in the weights for the " + self.data_type + " data"):
            with open(file, "rb") as np_file:
                data = np.load(np_file)
            n_elements = int(data.shape[0])
            weights[N : N + n_elements] = data[:, -3]
            N += n_elements
        return weights
