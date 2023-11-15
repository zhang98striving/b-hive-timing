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
import sys
from operator import itemgetter

from pympler import tracker

mem = tracker.SummaryTracker()


def merge_structured_arrays(array_list: list, delta: int = None):
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
    return merged, rest


def merge_datasets(files, path, label="", chunk_size=100000):
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
        disable=True,
    ) as progress:
        task = progress.add_task("Merging...", total=len(files))
        # # fmt: off
        # print(f"Entering debug in: {__file__}")
        # from IPython import embed;embed()
        # # fmt: on
        for i, file in enumerate(files):
            # If samples overflow chunk-size, write out new file
            d = {}
            with np.load(file, allow_pickle=True) as data:
                n_samples = len(data[data.files[0]])
                # # fmt: off
                # print(f"Entering debug in: {__file__}")
                # from IPython import embed;embed()
                # # fmt: on
                for field in data.files:
                    d[field] = data[field]
            merge_arrays.append(d)

            if i % 100 == 0:
                print(f"Iteration #{i}")
                print(f"opening file {file}")
                print("RAM memory % used:", psutil.virtual_memory()[2])
                # Getting usage of virtual_memory in GB ( 4th field)
                print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
                print(f"#n_samples\t{n_samples}")
                print(f"#n_chunk\t{n_chunk}")
                print(
                    sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10]
                )
                print(
                    "size of merged_arrays: {}".format(
                        sum(sys.getsizeof(m) for m in merge_arrays) / 1e9
                    )
                )
                print(
                    "size of d: {}".format(
                        sum(sys.getsizeof(dd) for dd in d.values()) / 1e9
                    )
                )
            while n_chunk + n_samples >= chunk_size:
                print("~-" * 20)
                print("merging")
                print("~-" * 20)
                merged, rest = merge_structured_arrays(
                    merge_arrays,
                    delta=chunk_size - n_chunk,
                )
                filename = os.path.join(path, f"{label}_{len(file_list)}.npz")
                file_list.append(filename)
                np.savez(filename, **merged)
                merge_arrays = [rest]
                n_chunk = 0
                n_samples = len(rest[list(rest.keys())[0]])
                n_chunk += n_samples

            progress.update(task, advance=1)
            progress.columns[-1].text_format = f"{i+1}/{len(files)} files merged"

    # writeout reamining arrays
    if len(merge_arrays) > 0:
        merged, _ = merge_structured_arrays(merge_arrays)
        filename = os.path.join(path, f"{label}_{len(file_list)}.npz")
        file_list.append(filename)
        np.savez(filename, **merged)

    return file_list
