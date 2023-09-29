import os
import numpy as np
from rich.progress import track


def merge_datasets(files, path, dim, label="", chunk_size=10000):
    file_index = 0
    n_chunk = 0
    file_list = []

    chunk = np.empty((chunk_size, dim), dtype=np.float32)

    for i, file in enumerate(track(files, "Merging...")):
        data = np.load(file, allow_pickle=True)
        n_samples = data.shape[0]
        # chunk overflow:
        if n_chunk + n_samples > chunk_size:
            index_range = chunk_size - n_chunk
        else:
            index_range = n_samples

        chunk[n_chunk : n_chunk + index_range] = data[:index_range]

        n_chunk += index_range
        if (n_chunk == chunk_size) or (i == len(files) - 1):
            filename = os.path.join(path, f"{label}_{file_index}.npy")
            file_list.append(filename)
            np.save(filename, chunk[:n_chunk])

            file_index += 1
            chunk = np.zeros((chunk_size, dim), dtype=np.float32)
            chunk[: n_samples - index_range] = data[index_range:]
            n_chunk = n_samples - index_range
    return file_list
