import os
from collections import defaultdict
from math import inf

import numpy as np
import numpy.lib.recfunctions as rfn
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
    files, path, label="", chunk_size=100000, verbose=0, shuffle=True, debug=False
):
    if shuffle:
        np.random.shuffle(files)
    n_chunk = 0
    file_list = []
    merge_arrays = []
    fields = np.load(files[0], allow_pickle=True, mmap_mode="r").files
    with Progress(
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn(f"0/{len(files)} files merged"),
        disable=True,  # for debugging, you might want to disable the merging progress bar
    ) as progress:
        task = progress.add_task("Merging...", total=len(files))
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
    # writeout reamining arrays
    if len(merge_arrays) > 0:
        if verbose:
            print("Merging remaining arrays")
        merged, _ = merge_structured_arrays(merge_arrays,shuffle=True)
        filename = os.path.join(path, f"{label}_{len(file_list)}.npz")
        file_list.append(filename)
        np.savez(filename, **merged)

    return file_list
