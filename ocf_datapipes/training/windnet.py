"""Create the training/validation datapipe for training the PVNet Model"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter import IterableWrapper

from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import (
    OpenConfiguration,
)
from ocf_datapipes.training.common import (
    _get_datapipes_dict,
    concat_xr_time_utc,
    construct_loctime_pipelines,
    fill_nans_in_arrays,
    fill_nans_in_pv,
    normalize_gsp,
    slice_datapipes_by_time,
)
from ocf_datapipes.utils.consts import (
    NEW_NWP_MEAN,
    NEW_NWP_STD,
    RSS_MEAN,
    RSS_STD,
)
from ocf_datapipes.utils.utils import combine_to_single_dataset, uncombine_from_single_dataset

xr.set_options(keep_attrs=True)
logger = logging.getLogger("windnet_datapipe")


def scale_wind_speed_to_power(x: Union[xr.DataArray, xr.Dataset]):
    """
    Scale wind speed to power to estimate the generation of wind power from ground sensors

    Roughly, double speed in m/s, and convert with the power scale

    Args:
        x: xr.DataArray or xr.Dataset containing wind speed

    Returns:
        Rescaled wind speed to MWh roughly
    """
    # m/s to kw (roughly) for each 1 m/s, starting from 0 to 30 m/s
    wind_speed_to_power = np.array(
        [
            0,
            0,
            0,
            0,
            66,
            171,
            352,
            623,
            1002,
            1497,
            2005,
            2246,
            2296,
        ]
    )
    # Convert knots to m/s
    x = x * 0.514444
    # Minimum speed is 0
    x = x.where(x > 0, 0)
    # Roughly double speed to get power
    # x = x * 2
    # convert to kw bsed on the wind_speed_to_power,
    # Do this by interpolating between the two nearest values in the list
    # Do this by rounding the wind speed to the nearest integer
    # x = x.round()
    # x = x.astype(int)
    # Convert to power for each element
    # x = xr.apply_ufunc(
    #    lambda x: wind_speed_to_power[x] if x < len(wind_speed_to_power) else 2296,
    #    x,
    #    vectorize=False,
    # )
    return x


def _normalize_wind_speed(x):
    return x / 30.0


@functional_datapipe("dict_datasets")
class DictDatasetIterDataPipe(IterDataPipe):
    """Create a dictionary of xr.Datasets from a set of iterators"""

    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe, keys: List[str]):
        """Init"""
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError(
                "All inputs are required to be `IterDataPipe` " "for `ZipIterDataPipe`."
            )
        super().__init__()
        self.keys = keys
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None
        assert len(self.keys) == len(self.datapipes), "Number of keys must match number of pipes"

    def __iter__(self):
        """Iter"""
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        for data in zip(*iterators):
            # Yield a dictionary of the data, using the keys in self.keys
            yield {k: v for k, v in zip(self.keys, data)}


@functional_datapipe("load_dict_datasets")
class LoadDictDatasetIterDataPipe(IterDataPipe):
    """Load NetCDF files and split them back into individual xr.Datasets"""

    filenames: List[str]
    keys: List[str]

    def __init__(self, filenames: List[str], keys: List[str]):
        """
        Load NetCDF files and split them back into individual xr.Datasets

        Args:
            filenames: List of filesnames to load
            keys: List of keys from each file to use, each key should be a
                dataarray in the xr.Dataset
        """
        super().__init__()
        self.keys = keys
        self.filenames = filenames

    def __iter__(self):
        """Iterate through each filename, loading it, uncombining it, and then yielding it"""
        while True:
            for filename in self.filenames:
                dataset = xr.open_dataset(filename)
                datasets = uncombine_from_single_dataset(dataset)
                # Yield a dictionary of the data, using the keys in self.keys
                dataset_dict = {}
                for k in self.keys:
                    dataset_dict[k] = datasets[k]
                yield dataset_dict


@functional_datapipe("convert_to_numpy_batch")
class ConvertToNumpyBatchIterDataPipe(IterDataPipe):
    """Converts Xarray Dataset to Numpy Batch"""

    def __init__(
        self,
        dataset_dict_dp: IterDataPipe,
        configuration: Configuration,
        check_satellite_no_zeros: bool = False,
    ):
        """Init"""
        super().__init__()
        self.dataset_dict_dp = dataset_dict_dp
        self.configuration = configuration
        self.check_satellite_no_zeros = check_satellite_no_zeros

    def __iter__(self):
        """Iter"""
        for datapipes_dict in self.dataset_dict_dp:
            # Spatially slice, normalize, and convert data to numpy arrays
            numpy_modalities = []
            # Unpack for convenience
            if "nwp" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["nwp"].convert_nwp_to_numpy_batch())
            if "sat" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["sat"].convert_satellite_to_numpy_batch())
            if "pv" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["pv"].convert_pv_to_numpy_batch())
            if "gsp" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["gsp"].convert_gsp_to_numpy_batch())
            if "sensor" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["sensor"].convert_sensor_to_numpy_batch())

            logger.debug("Combine all the data sources")
            combined_datapipe = MergeNumpyModalities(numpy_modalities)

            logger.info("Filtering out samples with no data")
            # if self.check_satellite_no_zeros:
            # in production we don't want any nans in the satellite data
            #    combined_datapipe = combined_datapipe.map(check_nans_in_satellite_data)

            combined_datapipe = combined_datapipe.map(fill_nans_in_arrays)

            yield next(iter(combined_datapipe))


def minutes(num_mins: int):
    """Timedelta of a number of minutes.

    Args:
        num_mins: Minutes timedelta.
    """
    return timedelta(minutes=num_mins)


def construct_sliced_data_pipeline(
    config_filename: str,
    location_pipe: IterDataPipe,
    t0_datapipe: IterDataPipe,
    production: bool = False,
) -> dict:
    """Constructs data pipeline for the input data config file.

    This yields samples from the location and time datapipes.

    Args:
        config_filename: Path to config file.
        location_pipe: Datapipe yielding locations.
        t0_datapipe: Datapipe yielding times.
        production: Whether constucting pipeline for production inference.
    """

    datapipes_dict = _get_datapipes_dict(
        config_filename,
        production=production,
    )

    configuration = datapipes_dict.pop("config")

    # Unpack for convenience
    conf_sat = configuration.input_data.satellite
    conf_nwp = configuration.input_data.nwp

    # Slice all of the datasets by time - this is an in-place operation
    slice_datapipes_by_time(datapipes_dict, t0_datapipe, configuration, production)

    if "nwp" in datapipes_dict:
        nwp_datapipe = datapipes_dict["nwp"]
        # Only need forks for if there are multiple other soruces, can just use one for now
        # location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
            location_pipe,  # TODO Fix when using satellite
            roi_height_pixels=conf_nwp.nwp_image_size_pixels_height,
            roi_width_pixels=conf_nwp.nwp_image_size_pixels_width,
        )
        nwp_datapipe = nwp_datapipe.normalize(mean=NEW_NWP_MEAN, std=NEW_NWP_STD)

    if "sat" in datapipes_dict:
        sat_datapipe = datapipes_dict["sat"]

        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
            location_pipe_copy,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
        )
        sat_datapipe = sat_datapipe.normalize(mean=RSS_MEAN, std=RSS_STD)

    if "sensor" in datapipes_dict:
        # Recombine Sensor arrays - see function doc for further explanation
        sensor_datapipe = (
            datapipes_dict["sensor"]
            .zip_ocf(datapipes_dict["sensor_future"])
            .map(concat_xr_time_utc)
        )
        sensor_datapipe = sensor_datapipe.map(scale_wind_speed_to_power)
        sensor_datapipe = sensor_datapipe.normalize(normalize_fn=_normalize_wind_speed)
        sensor_datapipe = sensor_datapipe.map(fill_nans_in_pv)

    finished_dataset_dict = {"config": configuration}
    # GSP always assumed to be in data
    if "gsp" in datapipes_dict:
        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        gsp_future_datapipe = datapipes_dict["gsp_future"]
        gsp_future_datapipe = gsp_future_datapipe.select_spatial_slice_meters(
            location_datapipe=location_pipe_copy,
            roi_height_meters=1,
            roi_width_meters=1,
            dim_name="gsp_id",
        )

        gsp_datapipe = datapipes_dict["gsp"]
        gsp_datapipe = gsp_datapipe.select_spatial_slice_meters(
            location_datapipe=location_pipe,
            roi_height_meters=1,
            roi_width_meters=1,
            dim_name="gsp_id",
        )

        # Recombine GSP arrays - see function doc for further explanation
        gsp_datapipe = gsp_datapipe.zip_ocf(gsp_future_datapipe).map(concat_xr_time_utc)
        gsp_datapipe = gsp_datapipe.normalize(normalize_fn=normalize_gsp)

        finished_dataset_dict["gsp"] = gsp_datapipe
    if "nwp" in datapipes_dict:
        finished_dataset_dict["nwp"] = nwp_datapipe
    if "sat" in datapipes_dict:
        finished_dataset_dict["sat"] = sat_datapipe
    if "sensor" in datapipes_dict:
        finished_dataset_dict["sensor"] = sensor_datapipe

    return finished_dataset_dict


def windnet_datapipe(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> IterDataPipe:
    """
    Construct windnet pipeline for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time at which a sample can be selected.
        end_time: Maximum time at which a sample can be selected.
    """
    logger.info("Constructing windnet pipeline")

    # Open datasets from the config and filter to useable location-time pairs
    location_pipe, t0_datapipe = construct_loctime_pipelines(
        config_filename,
        start_time,
        end_time,
    )

    # Shard after we have the loc-times. These are already shuffled so no need to shuffle again
    location_pipe = location_pipe.sharding_filter()
    t0_datapipe = t0_datapipe.sharding_filter()

    # In this function we re-open the datasets to make a clean separation before/after sharding
    # This function
    datapipe_dict = construct_sliced_data_pipeline(
        config_filename,
        location_pipe,
        t0_datapipe,
    )

    # Save out datapipe to NetCDF
    keys = list(datapipe_dict.keys())
    # Remove config
    keys.remove("config")

    # Merge all the datapipes into one
    return DictDatasetIterDataPipe(
        *[datapipe_dict[k] for k in keys],
        keys=keys,
    ).map(combine_to_single_dataset)


def split_dataset_dict_dp(element):
    """
    Split the dictionary of datapipes into individual datapipes

    Args:
        element: Dictionary of datapipes
    """
    return {k: IterableWrapper([v]) for k, v in element.items() if k != "config"}


def windnet_netcdf_datapipe(
    config_filename: str,
    keys: List[str],
    filenames: List[str],
) -> IterDataPipe:
    """
    Load the saved Datapipes from windnet, and transform to numpy batch

    Args:
        config_filename: Path to config file.
        keys: List of keys to extract from the single NetCDF files
        filenames: List of NetCDF files to load

    Returns:
        Datapipe that transforms the NetCDF files to numpy batch
    """
    logger.info("Constructing windnet file pipeline")
    config_datapipe = OpenConfiguration(config_filename)
    configuration: Configuration = next(iter(config_datapipe))
    # Load files
    datapipe_dict_dp: IterDataPipe = LoadDictDatasetIterDataPipe(
        filenames=filenames,
        keys=keys,
    ).map(split_dataset_dict_dp)
    datapipe = datapipe_dict_dp.convert_to_numpy_batch(configuration=configuration)

    return datapipe


if __name__ == "__main__":
    # Load the ECMWF and sensor data here
    datapipe = windnet_datapipe(
        config_filename="/home/jacob/Development/ocf_datapipes/tests/config/india_test.yaml",
        start_time=datetime(2021, 1, 1),
        end_time=datetime(2022, 1, 2),
    )
    batch = next(iter(datapipe))
    print(batch)
    batch.to_netcdf("test.nc", engine="h5netcdf")
    # Load the saved NetCDF files here
    datapipe = windnet_netcdf_datapipe(
        config_filename="/home/jacob/Development/ocf_datapipes/tests/config/india_test.yaml",
        keys=["nwp", "sensor"],
        filenames=["test.nc"],
    )
    batch = next(iter(datapipe))
    print(batch)
