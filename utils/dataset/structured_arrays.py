import awkward as ak
import numpy as np

from typing import List


def structured_array_from_tree(
    events=None, keys: list[str] = None, feature_length: int = None, precision=np.float32
) -> np.ndarray:
    dtype = np.dtype(
        [
            (name, precision, feature_length) if feature_length > 1 else (name, precision)
            for name in keys
        ]
    )
    arr = np.empty((len(events),), dtype=dtype)
    for key, dtype_name in zip(keys, dtype.fields):
        if feature_length == 1:
            arr[key] = np.array(events[key], dtype=[(dtype_name, precision)])
        else:
            arr[key] = ak.to_numpy(
                ak.values_astype(
                    ak.fill_none(ak.pad_none(events[key], feature_length)[:, :feature_length], 0),
                    np.float32,
                )
            )
    return arr
