import numpy as np
import torch
import lz4.frame

from functools import reduce
from numpy.lib import recfunctions
from rich.progress import track
from torch.utils.data import IterableDataset

class LZ4FP16Dataset(IterableDataset):
    def __init__(
        self,
        files,
        model,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        histogram_training=None,
        max_length=1,
        process_weights=None,
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
            with lz4.frame.open(self.files[0], mode='r') as fp:
                output_data = fp.read()
            s = np.frombuffer(output_data, dtype='float16')
            s = s[2:].reshape(-1, int(s[1]))
            self.all_number_of_samples = s.shape[0]

        self.weighted_sampling = weighted_sampling

        self.process_weights = process_weights

        self.device = device
        self.model = model

        self.num_truth = len(model.classes.keys())
        self.num_ele = 0

        for list_truth in model.classes:
            self.num_ele += len(model.classes[list_truth])

    def __len__(self):
        return int(self.all_number_of_samples*len(self.files))

    def __getitem__(self, index):
        raise NotImplementedError

    def shuffleFileList(self):
        np.random.shuffle(self.files)

    def __iter__(self):
        # Multi-worker support:
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = self.files
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[worker_info.id]

        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")

            #Add LZ4 loading
            with lz4.frame.open(file, mode='r') as data:
                output_data = data.read()
            s = np.frombuffer(output_data, dtype='float16')
            s = s[2:].reshape(-1, int(s[1])).astype('float16')

            s1 = ~np.isnan(s).any(axis = 1)
            s2 = ~np.isinf(s).any(axis = 1)
            s = s[s1*s2]

            if self.weighted_sampling:
                random_number = np.random.rand(s.shape[0])
                mask = random_number < s[:, -1]
                s = s[mask]

            if self.verbose:
                print(f"Keeping {np.sum(mask)}/{len(mask)} events")

            truths = np.zeros(s.shape[0])
            labels = s[:,-(self.num_ele+1):-1]
            idx = 0
            for index, (name, flavours) in enumerate(self.model.classes.items()):
                for flav in flavours:
                    truths[labels[:,idx] == 1] = index
                    idx += 1
            weights = s[:, -1]
            process = s[:, -(self.num_ele+2)]
            s = s[:,:-(self.num_ele+2)]

            for (si, yi, wi, pi) in zip(s, truths, weights, process):
                yield np.expand_dims(si, axis=-1), yi, wi, pi
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
