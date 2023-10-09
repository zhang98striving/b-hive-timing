import os
import numpy as np
from rich.progress import track
from collections import defaultdict
import numpy.lib.recfunctions as rfn


def merge_structured_arrays(array_list):
    merged = {}
    for key in array_list[0].keys():
        merged[key] = np.concatenate([a[key] for a in array_list])
    return merged


def merge_datasets(files, path, label="", chunk_size=10000):
    file_index = 0
    n_chunk = 0
    file_list = []
    merge_arrays = []

    for i, file in enumerate(track(files, "Merging...")):
        data = np.load(file, allow_pickle=True)
        n_samples = len(data[data.files[0]])

        # if samples would overflow chunk-size, write out new file
        if n_chunk + n_samples > chunk_size:
            merged = merge_structured_arrays(merge_arrays)
            filename = os.path.join(path, f"{label}_{file_index}.npz")
            file_list.append(filename)
            np.savez(filename, **merged)
            file_index += 1
            merge_arrays = []
            n_chunk = 0

        n_chunk += n_samples
        merge_arrays.append(data)

    # writeout reamining arrays
    if len(merge_arrays) > 0:
        merged = merge_structured_arrays(merge_arrays)
        filename = os.path.join(path, f"{label}_{file_index}.npz")
        file_list.append(filename)
        np.savez(filename, **merged)
        file_index += 1

    return file_list
