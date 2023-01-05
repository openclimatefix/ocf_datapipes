"""Various utilites that didn't fit elsewhere"""
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence, Sized, Tuple, Union

import fsspec.asyn
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pandas.core.dtypes.common import is_datetime64_dtype
from pathy import Pathy
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter.combining import T_co

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch

logger = logging.getLogger(__name__)


def return_system_indices_which_has_contiguous_nan(
    arr: np.ndarray, check_interval: int = 287
) -> np.ndarray:
    """This function return system indices

    Returns indexes of system id's in which if they have
    contigous 289 NaN's.

    Args:
        arr: Array of each system pvoutput values for a single day
        check_interval: time range intervals respectively

    """
    # Checking the shape of the input array
    # The array would be a 2d-array which consists of number of (time_utc, pv_system_id)
    number_of_systems = arr.shape[1]

    system_index_values_to_be_dropped = []
    for i in range(0, number_of_systems):

        # For each system id
        single_system_single_day_pv_values = arr[:, i]

        # This loop checks NaN in every element in the array and if the count of NaN
        # is equal to defined interval, it stores the index of the pv system
        mask = np.concatenate(([False], np.isnan(single_system_single_day_pv_values), [False]))
        if ~mask.any():
            continue
        else:
            idx = np.nonzero(mask[1:] != mask[:-1])[0]
            max_count = (idx[1::2] - idx[::2]).max()

        if max_count == check_interval:
            system_index_values_to_be_dropped.append(i)

    return system_index_values_to_be_dropped


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


# https://github.com/pytorch/data/issues/865
@functional_datapipe("zip_ocf")
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    """
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated
    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)),
        >>> IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """

    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        """Init"""
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError(
                "All inputs are required to be `IterDataPipe` " "for `ZipIterDataPipe`."
            )
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        """Iter"""
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        for data in zip(*iterators):
            yield data

    def __len__(self) -> int:
        """Len"""
        if self.length is not None:
            if self.length == -1:
                raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = min(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)


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
