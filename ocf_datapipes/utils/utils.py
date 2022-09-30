"""Various utilites that didn't fit elsewhere"""
import logging
from pathlib import Path
from typing import Sequence, Union

import fsspec.asyn
import numpy as np
import pandas as pd
import xarray as xr
from pathy import Pathy

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch

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
    return np.where(mask, nums, np.NaN)


def assert_num_dims(tensor, num_expected_dims: int) -> None:
    """
    Asserts the tensor shape is correct

    Args:
        tensor: Tensor to check
        num_expected_dims: Number of expected dims
    """
    assert len(tensor.shape) == num_expected_dims, (
        f"Expected tensor to have {num_expected_dims} dims." f" Instead, shape={tensor.shape}"
    )


def is_sorted(array: np.ndarray) -> bool:
    """Return True if array is sorted in ascending order."""
    # Adapted from https://stackoverflow.com/a/47004507/732596
    if len(array) == 0:
        return False
    return np.all(array[:-1] <= array[1:])


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


def sample_row_and_drop_row_from_df(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.Series, pd.DataFrame]:
    """Return sampled_row, dataframe_with_row_dropped."""
    assert not df.empty
    row_idx = rng.integers(low=0, high=len(df))
    row = df.iloc[row_idx]
    df = df.drop(row.name)
    return row, df


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


def stack_np_examples_into_batch(np_examples: Sequence[NumpyBatch]) -> NumpyBatch:
    """
    Stacks Numpy examples into a batch

    Args:
        np_examples: Numpy examples to stack

    Returns:
        The stacked NumpyBatch object
    """
    np_batch: NumpyBatch = {}
    batch_keys = np_examples[0]  # Batch keys should be the same across all examples.
    for batch_key in batch_keys:
        if batch_key.name.endswith("t0_idx") or batch_key == BatchKey.nwp_channel_names:
            # These are always the same for all examples.
            np_batch[batch_key] = np_examples[0][batch_key]
        else:
            examples_for_key = [np_example[batch_key] for np_example in np_examples]
            try:
                np_batch[batch_key] = np.stack(examples_for_key)
            except Exception as e:
                logger.debug(f"Could not stack the following shapes together, ({batch_key})")
                shapes = [example_for_key.shape for example_for_key in examples_for_key]
                logger.debug(shapes)
                logger.error(e)
                raise e
    return np_batch


def select_time_periods(
    xr_data: Union[xr.DataArray, xr.Dataset], time_periods: pd.DataFrame, dim_name: str = "time_utc"
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Selects time periods from Xarray object

    Args:
        xr_data: Xarray object
        time_periods: Time periods to select
        dim_name: Dimension name for time

    Returns:
        The subselected Xarray object
    """
    new_xr_data = []
    for _, row in time_periods.iterrows():
        start_dt = row["start_dt"]
        end_dt = row["end_dt"]
        new_xr_data.append(xr_data.sel({dim_name: slice(start_dt, end_dt)}))
    return xr.concat(new_xr_data, dim=dim_name)


def pandas_periods_to_our_periods_dt(
    periods: Union[Sequence[pd.Period], pd.PeriodIndex]
) -> pd.DataFrame:
    """
    Converts Pandas periods to new periods

    Args:
        periods: Pandas periods to convert

    Returns:
        Converted pandas periods
    """
    new_periods = []
    for period in periods:
        new_periods.append(dict(start_dt=period.start_time, end_dt=period.end_time))
    return pd.DataFrame(new_periods)
