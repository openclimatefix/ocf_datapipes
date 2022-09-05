import xarray as xr
import  numpy as np
import pandas as pd
from numbers import Number

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils.consts import SAT_VARIABLE_NAMES, NWP_VARIABLE_NAMES
from typing import Sequence, Optional
from ocf_datapipes.fake.coordinates import make_image_coords_osgb, make_random_x_and_y_osgb_centers, create_random_point_coordinates_osgb
from ocf_datapipes.utils.geospatial import osgb_to_lat_lon
from datetime import datetime


def make_t0_datetimes_utc(batch_size, temporally_align_examples: bool = False):
    """
    Make list of t0 datetimes

    Args:
        batch_size: the batch size
        temporally_align_examples: option to align examples (within the batch) in time

    Returns: pandas index of t0 datetimes
    """

    all_datetimes = pd.date_range("2021-01-01", "2021-02-01", freq="5T")

    if temporally_align_examples:
        t0_datetimes_utc = list(np.random.choice(all_datetimes, size=1)) * batch_size
    else:
        if len(all_datetimes) >= batch_size:
            replace = False
        else:
            # there are not enought data points,
            # so some examples will have the same datetime
            replace = True

        t0_datetimes_utc = np.random.choice(all_datetimes, batch_size, replace=replace)
    # np.random.choice turns the pd.Timestamp objects into datetime.datetime objects.

    t0_datetimes_utc = pd.to_datetime(t0_datetimes_utc)

    # TODO make test repeatable using numpy generator
    # https://github.com/openclimatefix/nowcasting_dataset/issues/594

    return t0_datetimes_utc


def create_image_array(
    nwp_or_satellite: str = "nwp",
    seq_length: int = 19,
    history_seq_length: int = 5,
    image_size_pixels_height: int = 64,
    image_size_pixels_width: int = 64,
    channels: Sequence[str] = SAT_VARIABLE_NAMES,
    freq: str = "5T",
    t0_datetime_utc: Optional[datetime] = None,
    x_center_osgb: Optional[Number] = None,
    y_center_osgb: Optional[Number] = None,
) -> xr.DataArray:
    """Create Satellite or NWP fake image data."""
    if t0_datetime_utc is None:
        t0_datetime_utc = make_t0_datetimes_utc(batch_size=1)[0]

    # We want the OSGB coords to be 2D for satellite data:
    two_dimensional_osgb_coords = nwp_or_satellite == "satellite"

    # Get OSB coords:
    x_osgb, y_osgb = make_image_coords_osgb(
        size_y=image_size_pixels_height,
        size_x=image_size_pixels_width,
        x_center_osgb=x_center_osgb,
        y_center_osgb=y_center_osgb,
        two_dimensional=two_dimensional_osgb_coords,
    )

    time = pd.date_range(end=t0_datetime_utc, freq=freq, periods=history_seq_length + 1).union(
        pd.date_range(start=t0_datetime_utc, freq=freq, periods=seq_length - history_seq_length)
    )

    # First, define coords which are common between NWP and satellite.
    # (Don't worry about the order of the dims. That will be defined using the `dims` arg
    # to the `xr.DataArray` constructor.)
    coords = {"time": time, "channels": np.array(channels)}

    # Now define coords and dims specific to nwp or satellite.
    if nwp_or_satellite == "nwp":
        dims = ("time", "y_osgb", "x_osgb", "channel")
        coords["y_osgb"] = ("y_osgb", y_osgb)
        coords["x_osgb"] = ("x_osgb", x_osgb)
    elif nwp_or_satellite == "satellite":
        dims = ("time", "y_geostationary", "x_geostationary", "channel")
        coords["y_osgb"] = (("y_geostationary", "x_geostationary"), y_osgb)
        coords["x_osgb"] = (("y_geostationary", "x_geostationary"), x_osgb)

        # Compute fake geostationary coords by converting the OSGB coords to geostationary,
        # and then selecting one row or one column from the converted coords (because the
        # geostationary coords are each 1D). Remember that `x_osgb` and `y_osgb` are
        # both 2D, of shape (y, x).
        x_geostationary, y_geostationary = osgb_to_geostationary(x=x_osgb, y=y_osgb)
        coords["y_geostationary"] = y_geostationary[:, 0]
        coords["x_geostationary"] = x_geostationary[0, :]
    else:
        raise ValueError(
            f"nwp_or_satellite must be either 'nwp' or 'satellite', not '{nwp_or_satellite}'"
        )

    image_data_array = xr.DataArray(
        data=abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(seq_length, image_size_pixels_height, image_size_pixels_width, len(channels)),
            )
        ),
        dims=dims,
        coords=coords,
        name="data",
    )  # Fake data for testing!

    return image_data_array


def create_gsp_pv_dataset(
    dims=("time", "id"),
    freq="5T",
    seq_length=19,
    history_seq_length=5,
    number_of_systems=128,
    time_dependent_capacity: bool = True,
    t0_datetime_utc: Optional = None,
    x_center_osgb: Optional = None,
    y_center_osgb: Optional = None,
    id_limit: int = 2048,
) -> xr.Dataset:
    """
    Create gsp or pv fake dataset

    Args:
        dims: the dims that are made for "power_mw"
        freq: the frequency of the time steps
        seq_length: the time sequence length
        number_of_systems: number of pv or gsp systems
        time_dependent_capacity: if the capacity is time dependent.
            GSP capacities increase over time,
            but PV systems are the same (or should be).
        history_seq_length: The historic length
        t0_datetime_utc: the time now, if this is not given, a random one will be made.
        x_center_osgb: the x center of the example. If not given, a random one will be made.
        y_center_osgb: the y center of the example. If not given, a random one will be made.
        id_limit: The maximum id number allowed. For example for GSP it should be 338

    Returns: xr.Dataset of fake data

    """

    if t0_datetime_utc is None:
        t0_datetime_utc = make_t0_datetimes_utc(batch_size=1)[0]

    time = pd.date_range(end=t0_datetime_utc, freq=freq, periods=history_seq_length + 1).union(
        pd.date_range(start=t0_datetime_utc, freq=freq, periods=seq_length - history_seq_length)
    )

    ALL_COORDS = {
        "time": time,
        "id": np.random.choice(range(id_limit), number_of_systems, replace=False),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]

    # make pv yield.  randn samples from a Normal distribution (and so can go negative).
    # The values are clipped to be positive later.
    data = np.random.randn(seq_length, number_of_systems)

    # smooth the data, the convolution method smooths that data across systems first,
    # and then a bit across time (depending what you set N)
    N = int(seq_length / 2)
    data = np.convolve(data.ravel(), np.ones(N) / N, mode="same").reshape(
        (seq_length, number_of_systems)
    )
    # Need to clip  *after* smoothing, because the smoothing method might push
    # non-zero data below zero.  Clip at 0.1 instead of 0 so we don't get div-by-zero errors
    # if capacity is zero (capacity is computed as the max of the random numbers).
    data = data.clip(min=0.1)

    # make into a Data Array
    data_array = xr.DataArray(
        data,
        coords=coords,
    )  # Fake data for testing!

    capacity = data_array.max(dim="time")
    if time_dependent_capacity:
        capacity = capacity.expand_dims(time=seq_length)
        capacity.__setitem__("time", data_array.time.values)

    data = data_array.to_dataset(name="power_mw")

    # make random coords
    x, y = create_random_point_coordinates_osgb(
        size=number_of_systems, x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb
    )

    x_coords = xr.DataArray(
        data=x,
        dims=["id"],
    )

    y_coords = xr.DataArray(
        data=y,
        dims=["id"],
    )

    data["capacity_mwp"] = capacity
    data["x_osgb"] = x_coords
    data["y_osgb"] = y_coords

    # Add 1000 to the id numbers for the row numbers.
    # This is a quick way to make sure row number is different from id,
    data["pv_system_row_number"] = data["id"] + 1000

    data.__setitem__("power_mw", data.power_mw.clip(min=0))

    return data
