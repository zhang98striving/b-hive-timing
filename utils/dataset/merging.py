import os
from collections import defaultdict
from math import inf
import lz4.frame

import numpy as np
import numpy.lib.recfunctions as rfn
from utils.weighting.histogram import histogram_weighting

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# debug
import psutil


def merge_structured_arrays(array_list: list, delta: int = None, shuffle: bool = True):
    merged = {}
    rest = {}
    # Start merging into traget array:
    for key in array_list[0].keys():
        # merge all arrays up the very last one, if there are multiple files
        if len(array_list) > 1:
            merged[key] = np.concatenate([a[key] for a in array_list[:-1]])
            # merge the last one partially:
            merged[key] = np.concatenate([merged[key], array_list[-1][key][:delta]])
        # if there is only one file, slice it by delta
        else:
            merged[key] = array_list[-1][key][:delta]
        # keep the last chunk (overflow)
        rest[key] = array_list[-1][key][delta:]

    if shuffle:
        indices = np.arange(len(merged[key]))
        np.random.shuffle(indices)
        for key in merged.keys():
            merged[key] = merged[key][indices]

    return merged, rest


def check_memory_usage():
    memory_usage = psutil.virtual_memory().used / (1024.0**3)
    return memory_usage


def merge_datasets(
        files, path, label="", chunk_size=100000, verbose=0, processor = "", shuffle=True, debug=False, histograms = None, reference_key= None, bins_pt=None, bins_eta=None,
):
    if shuffle:
        np.random.shuffle(files)
    n_chunk = 0
    file_list = []
    merge_arrays = []
    with Progress(
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn(f"0/{len(files)} files merged"),
    ) as progress:
        task = progress.add_task("Merging...", total=len(files))
        print(processor)
        if (processor == "LZ4Processing") or (processor == "LZ4FP16Processing"):
            if processor == "LZ4FP16Processing":
                dtype = np.float16
            else:
                dtype = np.float32
            dim = np.load(files[0][:-4]+'.npy', allow_pickle=True).shape[-1]
            chunk = np.empty((chunk_size, dim), dtype=dtype)

            reference_histogram = histograms[0]
            reference_histogram = reference_histogram / np.max(reference_histogram)
            weights_list = []
            for c in range(histograms.shape[0]):
                other_histogram = histograms[c]
                other_histogram = other_histogram / np.max(other_histogram)
                with np.errstate(divide="ignore", invalid="ignore"):
                    weights = np.where(other_histogram > 0, reference_histogram / other_histogram, -10)
                weights = weights / np.max(weights)
                
                weights[weights < 0] = 1
                weights[weights == np.nan] = 1
                
                weights_list.append(weights)
                
            weights_list = np.array(weights_list)
            
            for i, file in enumerate(files):
                data = np.load(file[:-4]+'.npy', allow_pickle=True)
                n_samples = len(data)

                if n_chunk + n_samples > chunk_size:
                    index_range = chunk_size - n_chunk
                else:
                    index_range = n_samples

                chunk[n_chunk : n_chunk + index_range] = data[:index_range]
                n_chunk += index_range
                
                if n_chunk >= chunk_size:
                    filename = os.path.join(path, f"{label}_{len(file_list)}.lz4")
                    file_list.append(filename)

                    s1 = ~np.isnan(chunk).any(axis = 1)
                    s2 = ~np.isinf(chunk).any(axis = 1)
                    chunk = chunk[s1*s2]

                    pt_coordinate = np.digitize(chunk[:,0], bins_pt) - 1
                    eta_coordinate = np.digitize(chunk[:,1], bins_eta) - 1
                    flavour_idx = np.argmax(chunk[:,-histograms.shape[0]:], axis=-1)

                    w = weights_list[flavour_idx, pt_coordinate, eta_coordinate].astype(dtype)
                    chunk = np.concatenate((chunk,w.reshape(-1,1)), axis=1)
                    size = np.array(chunk.shape).astype(dtype)
                    
                    arr = np.concatenate((size, chunk.astype(dtype).flatten()))
                    arr = arr.tobytes()
                    print(filename)
                    with lz4.frame.open(filename, mode='wb') as fp:
                        bytes_written = fp.write(arr)

                    chunk = np.zeros((chunk_size, dim), dtype=dtype)
                    chunk[: n_samples - index_range] = data[index_range:]
                    n_chunk = n_samples - index_range
            for i, file in enumerate(files):
                os.remove(file[:-4]+'.npy')
            
        else:
            fields = np.load(files[0], allow_pickle=True, mmap_mode="r").files
            for i, file in enumerate(files):
                with np.load(file, allow_pickle=True) as data:
                    n_samples = len(data[data.files[0]])
                    merge_arrays.append(dict(data))
                    
                # Check memory usage
                if debug:
                    if i % 100 == 0:
                        memory_usage = check_memory_usage()
                        print(
                            "Current memory usage: {0:1.2f} GB; arrays: {1:4d}; size of one element: {2:1.2f}MB;n_samples: {3:8d}/{4:8d} - {5:2.1f}%".format(
                                memory_usage,
                                len(merge_arrays),
                                list(merge_arrays[0].values())[0].size
                                * list(merge_arrays[0].values())[0].itemsize
                                / 1e6,
                                n_chunk,
                                chunk_size,
                                n_chunk / chunk_size * 100,
                            )
                        )
                # If samples overflow chunk-size, write out new file
                while n_chunk + n_samples >= chunk_size:
                    if verbose:
                        print("Merging remaining arrays")
                    merged, rest = merge_structured_arrays(
                        merge_arrays,
                        delta=chunk_size - n_chunk,
                        shuffle=True,
                    )
                    filename = os.path.join(path, f"{label}_{len(file_list)}.npz")
                    file_list.append(filename)
                    np.savez(filename, **merged)
                    merge_arrays.clear()
                    merge_arrays = [rest]
                    del merged, rest
                    n_chunk = 0
                    n_samples = len(merge_arrays[0][fields[0]])
                n_chunk += n_samples

                progress.update(task, advance=1)
                progress.columns[-1].text_format = f"{i+1}/{len(files)} files merged"
                # writeout remaining arrays
            if len(merge_arrays) > 0:
                if verbose:
                    print("Merging remaining arrays")
                merged, _ = merge_structured_arrays(merge_arrays,shuffle=True)
                filename = os.path.join(path, f"{label}_{len(file_list)}.npz")
                file_list.append(filename)
                np.savez(filename, **merged)
            
    return file_list
