"""Various utilites that didn't fit elsewhere"""

import logging
from pathlib import Path
from typing import Tuple, Union

import fsspec.asyn
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pandas.core.dtypes.common import is_datetime64_dtype
from pathy import Pathy

logger = logging.getLogger(__name__)


def datetime64_to_float(datetimes: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Converts datetime64 to floats

    Args:
        datetimes: Array of datetimes
        dtype: Dtype to convert to

    Returns:
        Converted datetimes to floats
    """
    nums = datetimes.astype("datetime64[s]").astype(dtype)
    mask = np.isfinite(datetimes)
    return np.where(mask, nums, np.nan)


def is_sorted(array: np.ndarray) -> bool:
    """Return True if array is sorted in ascending order."""
    # Adapted from https://stackoverflow.com/a/47004507/732596
    if len(array) == 0:
        return False
    return np.all(array[:-1] <= array[1:])


def searchsorted(a, v, side="left", assume_ascending=True):
    """Find indices where elements should be inserted to maintain order.

    Can be either sorted in ascending order or descending order

    Args:
        a: Target array
        v: Values to insert into `a`
        side: If ‘left’, the index of the first suitable location found is given. If ‘right’, return
            the last such index
        assume_ascending: Whether `a` is in ascending order, else assumed descending
    """
    if assume_ascending:
        return np.searchsorted(a, v, side=side)
    else:
        return np.searchsorted(-a, -v, side=side)


def check_path_exists(path: Union[str, Path]):
    """Raise a FileNotFoundError if `path` does not exist.

    `path` can include wildcards.
    """
    if not path:
        raise FileNotFoundError("Not a valid path!")
    filesystem = get_filesystem(path)
    if not filesystem.exists(path):
        # Maybe `path` includes a wildcard? So let's use `glob` to check.
        # Try `exists` before `glob` because `glob` might be slower.
        files = filesystem.glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f"{path} does not exist!")


def get_filesystem(path: Union[str, Path]) -> fsspec.AbstractFileSystem:
    r"""Get the fsspect FileSystem from a path.

    For example, if `path` starts with `gs://` then return a gcsfs.GCSFileSystem.

    It is safe for `path` to include wildcards in the final filename.
    """
    path = Pathy(path)
    return fsspec.open(path.parent).fs


def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.

    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!

    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0

    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948

    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None
    fsspec.asyn._lock = None


def _trig_transform(values: np.ndarray, period: Union[float, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of values and an upper limit on the values, compute trig decomposition.

    Args:
        values: ndarray of points in the range [0, period]
        period: period of the data

    Returns:
        Decomposition of values into sine and cosine of data with given period
    """

    return np.sin(values * 2 * np.pi / period), np.cos(values * 2 * np.pi / period)


def trigonometric_datetime_transformation(datetimes: npt.ArrayLike) -> np.ndarray:
    """
    Given an iterable of datetimes, returns a trigonometric decomposition on hour, day and month

    Args:
        datetimes: ArrayLike of datetime64 values

    Returns:
        Trigonometric decomposition of datetime into hourly, daily and
        monthly values.
    """
    assert is_datetime64_dtype(datetimes), "Data for Trig Decomposition must be np.datetime64 type"

    datetimes = pd.DatetimeIndex(datetimes)
    hour = datetimes.hour.values.reshape(-1, 1) + (datetimes.minute.values.reshape(-1, 1) / 60)
    day = datetimes.day.values.reshape(-1, 1)
    month = datetimes.month.values.reshape(-1, 1)

    sine_hour, cosine_hour = _trig_transform(hour, 24)
    sine_day, cosine_day = _trig_transform(day, 366)
    sine_month, cosine_month = _trig_transform(month, 12)

    return np.concatenate(
        [sine_month, cosine_month, sine_day, cosine_day, sine_hour, cosine_hour], axis=1
    )


def combine_to_single_dataset(dataset_dict: dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Combine multiple datasets into a single dataset

    Args:
        dataset_dict: Dictionary of xr.Dataset objects to combine

    Returns:
        Combined dataset
    """
    # Flatten any NWP data
    dataset_dict = flatten_nwp_source_dict(dataset_dict, sep="-")

    # Convert all data_arrays to datasets
    new_dataset_dict = {}
    for key, datasets in dataset_dict.items():
        new_datasets = []
        for dataset in datasets:
            # Convert all coordinates float64 and int64 to float32 and int32
            dataset = dataset.assign_attrs(
                {key: str(value) for key, value in dataset.attrs.items()}
            )
            if isinstance(dataset, xr.DataArray):
                new_datasets.append(dataset.to_dataset(name=key))
            else:
                new_datasets.append(dataset)
            assert isinstance(new_datasets[-1], xr.Dataset)
        new_dataset_dict[key] = new_datasets

    # Prepend all coordinates and dimensions names with the key in the dataset_dict
    final_datasets_to_combined = []
    for key, datasets in new_dataset_dict.items():
        batched_datasets = []
        for dataset in datasets:
            dataset = dataset.rename(
                {dim: f"{key}__{dim}" for dim in dataset.dims if dim not in dataset.coords}
            )
            dataset = dataset.rename({coord: f"{key}__{coord}" for coord in dataset.coords})
            batched_datasets.append(dataset)
        # Merge all datasets with the same key
        # If NWP, then has init_time_utc and step, so do it off key__init_time_utc
        dataset = xr.concat(
            batched_datasets,
            dim=(
                f"{key}__target_time_utc"
                if f"{key}__target_time_utc" in dataset.coords
                else f"{key}__time_utc"
            ),
        )

        # Make sure the init_time is a vector.
        # If they are all the same value,
        # then the concat reduces them down to a scalar
        key_init = f"{key}__init_time_utc"
        key_target_time = f"{key}__target_time_utc"
        if key_init in dataset.coords:
            # check to see if init_time_utc is a scalar
            if len(dataset[key_init].dims) == 0:
                # expand the init_time_utc to the same length as the target_time_utc
                init_time_utcs = [dataset[key_init].values] * len(dataset[key_target_time].values)
                dataset = dataset.assign_coords({key_init: (key_target_time, init_time_utcs)})

        # Serialize attributes to be JSON-seriaizable
        final_datasets_to_combined.append(dataset)
    # Combine all datasets, and append the list of datasets to the dataset_dict
    for f_dset in final_datasets_to_combined:
        assert isinstance(f_dset, xr.Dataset), f"Dataset is not an xr.Dataset, {type(f_dset)}"
    combined_dataset = xr.merge(final_datasets_to_combined)
    # Print all attrbutes of the combined dataset
    return combined_dataset


def uncombine_from_single_dataset(combined_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Uncombine a combined dataset

    Args:
        combined_dataset: The combined NetCDF dataset

    Returns:
        The uncombined datasets as a dict of xr.Datasets
    """
    # Split into datasets by splitting by the prefix added in combine_to_netcdf
    datasets = {}
    # Go through each data variable and split it into a dataset
    for key, dataset in combined_dataset.items():
        # If 'key_' doesn't exist in a dim or coordinate, remove it
        dataset_dims = list(dataset.coords)
        for dim in dataset_dims:
            if f"{key}__" not in dim:
                dataset: xr.DataArray = dataset.drop(dim)
        dataset = dataset.rename(
            {dim: dim.split(f"{key}__")[1] for dim in dataset.dims if dim not in dataset.coords}
        )
        dataset: xr.Dataset = dataset.rename(
            {coord: coord.split(f"{key}__")[1] for coord in dataset.coords}
        )
        # Split the dataset by the prefix
        datasets[key] = dataset

    # Unflatten any NWP data
    datasets = nest_nwp_source_dict(datasets, sep="-")
    return datasets


def flatten_nwp_source_dict(d: dict, sep: str = "/") -> dict:
    """Unnest a dictionary where the NWP values are nested under the key 'nwp'."""
    new_dict = {k: v for k, v in d.items() if k != "nwp"}
    if "nwp" in d:
        if isinstance(d["nwp"], dict):
            new_dict.update({f"nwp{sep}{k}": v for k, v in d["nwp"].items()})
        else:
            new_dict.update({"nwp": d["nwp"]})
    return new_dict


def nest_nwp_source_dict(d: dict, sep: str = "/") -> dict:
    """Re-nest a dictionary where the NWP values are nested under keys 'nwp/<key>'."""
    nwp_prefix = f"nwp{sep}"
    new_dict = {k: v for k, v in d.items() if not k.startswith(nwp_prefix)}
    nwp_keys = [k for k in d.keys() if k.startswith(nwp_prefix)]
    if len(nwp_keys) > 0:
        nwp_subdict = {k.removeprefix(nwp_prefix): d[k] for k in nwp_keys}
        new_dict["nwp"] = nwp_subdict
    return new_dict
