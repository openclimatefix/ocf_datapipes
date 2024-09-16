"""Create the training/validation datapipe for training the PVNet Model"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter import IterableWrapper

from ocf_datapipes.batch import BatchKey, MergeNumpyModalities, MergeNWPNumpyModalities
from ocf_datapipes.training.common import (
    DatapipeKeyForker,
    _get_datapipes_dict,
    concat_xr_time_utc,
    construct_loctime_pipelines,
    fill_nans_in_arrays,
    fill_nans_in_pv,
    normalize_gsp,
    potentially_coarsen,
    slice_datapipes_by_time,
)
from ocf_datapipes.utils.consts import (
    NWP_MEANS,
    NWP_STDS,
    RSS_MEAN,
    RSS_RAW_MAX,
    RSS_RAW_MIN,
    RSS_STD,
)
from ocf_datapipes.utils.utils import (
    combine_to_single_dataset,
    flatten_nwp_source_dict,
    nest_nwp_source_dict,
    uncombine_from_single_dataset,
)

xr.set_options(keep_attrs=True)
logger = logging.getLogger("pvnet_site_datapipe")


def normalize_pv(x: xr.DataArray):
    """Normalize PV data"""
    return x / x.nominal_capacity_wp


class DictDatasetIterDataPipe(IterDataPipe):
    """Create a dictionary of xr.Datasets from a dict of datapipes"""

    datapipes_dict: dict[IterDataPipe]
    length: Optional[int]

    def __init__(self, datapipes_dict: dict[IterDataPipe]):
        """Init"""
        # Flatten the dict of input datapipes (NWP is nested)
        self.datapipes_dict = flatten_nwp_source_dict(datapipes_dict)
        self.length = None

        # Run checks
        is_okay = all([isinstance(dp, IterDataPipe) for k, dp in self.datapipes_dict.items()])

        if not is_okay:
            raise TypeError(
                "All inputs are required to be `IterDataPipe` " "for `ZipIterDataPipe`."
            )

        super().__init__()

    def __iter__(self):
        """Iter"""
        all_keys = []
        all_datapipes = []
        for k, dp in self.datapipes_dict.items():
            all_keys += [k]
            all_datapipes += [dp]

        zipped_datapipes = all_datapipes[0].zip_ocf(*all_datapipes[1:])

        for values in zipped_datapipes:
            output_dict = {key: x for key, x in zip(all_keys, values)}

            # re-nest the nwp keys
            output_dict = nest_nwp_source_dict(output_dict)

            yield output_dict


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
                datasets["nwp"]["ecmwf"] = potentially_coarsen(datasets["nwp"]["ecmwf"])
                # Yield a dictionary of the data, using the keys in self.keys
                dataset_dict = {}
                for k in self.keys:
                    dataset_dict[k] = datasets[k]
                yield dataset_dict


@functional_datapipe("pvnet_site_convert_to_numpy_batch")
class ConvertToNumpyBatchIterDataPipe(IterDataPipe):
    """Converts Xarray Dataset to Numpy Batch"""

    def __init__(
        self,
        dataset_dict_dp: IterDataPipe,
        check_satellite_no_zeros: bool = False,
    ):
        """Init"""
        super().__init__()
        self.dataset_dict_dp = dataset_dict_dp
        self.check_satellite_no_zeros = check_satellite_no_zeros

    def __iter__(self):
        """Iter"""
        for datapipes_dict in self.dataset_dict_dp:
            # Spatially slice, normalize, and convert data to numpy arrays
            numpy_modalities = []

            if "nwp" in datapipes_dict:
                # Combine the NWPs into NumpyBatch
                nwp_numpy_modalities = dict()
                for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():
                    nwp_numpy_modalities[nwp_key] = nwp_datapipe.convert_nwp_to_numpy_batch()

                nwp_numpy_modalities = MergeNWPNumpyModalities(nwp_numpy_modalities)
                numpy_modalities.append(nwp_numpy_modalities)

            if "sat" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["sat"].convert_satellite_to_numpy_batch())
            if "pv" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["pv"].convert_pv_to_numpy_batch())
            if "gsp" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["gsp"].convert_gsp_to_numpy_batch())
            if "sensor" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["sensor"].convert_sensor_to_numpy_batch())
            if "wind" in datapipes_dict:
                numpy_modalities.append(datapipes_dict["wind"].convert_wind_to_numpy_batch())

            logger.debug("Combine all the data sources")
            combined_datapipe = MergeNumpyModalities(numpy_modalities).add_sun_position(
                modality_name="pv"
            )

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

    # We need a copy of the location datapipe for all keys in fork_keys
    fork_keys = set(k for k in datapipes_dict.keys())
    if "nwp" in datapipes_dict:  # NWP is nested
        fork_keys.update(set(f"nwp/{k}" for k in datapipes_dict["nwp"].keys()))

    # We don't need somes keys even if they are in the data dictionary
    fork_keys = fork_keys - set(
        ["topo", "nwp", "wind", "wind_future", "sensor", "hrv", "pv_future", "pv"]
    )

    # Set up a key-forker for all the data sources we need it for
    get_loc_datapipe = DatapipeKeyForker(fork_keys, location_pipe)

    if "nwp" in datapipes_dict:
        nwp_datapipes_dict = dict()

        for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():
            location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
            nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
                get_loc_datapipe(f"nwp/{nwp_key}"),
                roi_height_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_height,
                roi_width_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_width,
            )
            # Coarsen the data, if it is separated by 0.05 degrees each
            nwp_datapipe = nwp_datapipe.map(potentially_coarsen)
            # Somewhat hacky way for India specifically, need different mean/std for ECMWF data
            if conf_nwp[nwp_key].nwp_provider in ["ecmwf"]:
                normalize_provider = "ecmwf_india"
            else:
                normalize_provider = conf_nwp[nwp_key].nwp_provider
            nwp_datapipes_dict[nwp_key] = nwp_datapipe.normalize(
                mean=NWP_MEANS[normalize_provider],
                std=NWP_STDS[normalize_provider],
            )

    if "sat" in datapipes_dict:
        sat_datapipe = datapipes_dict["sat"]

        sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
            get_loc_datapipe("sat"),
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
        )
        scaling_methods = conf_sat.satellite_scaling_methods
        if "min_max" in scaling_methods:
            sat_datapipe = sat_datapipe.normalize(min_values=RSS_RAW_MIN, max_values=RSS_RAW_MAX)
        if "mean_std" in scaling_methods:
            sat_datapipe = sat_datapipe.normalize(mean=RSS_MEAN, std=RSS_STD)

    if "pv" in datapipes_dict:
        # Recombine Sensor arrays - see function doc for further explanation
        pv_datapipe = (
            datapipes_dict["pv"].zip_ocf(datapipes_dict["pv_future"]).map(concat_xr_time_utc)
        )
        pv_datapipe = pv_datapipe.normalize(normalize_fn=normalize_pv)
        pv_datapipe = pv_datapipe.map(fill_nans_in_pv)

    finished_dataset_dict = {"config": configuration}

    if "gsp" in datapipes_dict:
        gsp_future_datapipe = datapipes_dict["gsp_future"]
        gsp_future_datapipe = gsp_future_datapipe.select_spatial_slice_meters(
            location_datapipe=get_loc_datapipe("gsp_future"),
            roi_height_meters=1,
            roi_width_meters=1,
            dim_name="gsp_id",
        )

        gsp_datapipe = datapipes_dict["gsp"]
        gsp_datapipe = gsp_datapipe.select_spatial_slice_meters(
            location_datapipe=get_loc_datapipe("gsp"),
            roi_height_meters=1,
            roi_width_meters=1,
            dim_name="gsp_id",
        )

        # Recombine GSP arrays - see function doc for further explanation
        gsp_datapipe = gsp_datapipe.zip_ocf(gsp_future_datapipe).map(concat_xr_time_utc)
        gsp_datapipe = gsp_datapipe.normalize(normalize_fn=normalize_gsp)

        finished_dataset_dict["gsp"] = gsp_datapipe

    get_loc_datapipe.close()

    if "nwp" in datapipes_dict:
        finished_dataset_dict["nwp"] = nwp_datapipes_dict
    if "sat" in datapipes_dict:
        finished_dataset_dict["sat"] = sat_datapipe
    if "pv" in datapipes_dict:
        finished_dataset_dict["pv"] = pv_datapipe

    return finished_dataset_dict


def pvnet_site_datapipe(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> IterDataPipe:
    """
    Construct PVNet site pipeline for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time at which a sample can be selected.
        end_time: Maximum time at which a sample can be selected.
    """
    logger.info("Constructing pvnet site pipeline")

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

    # Merge all the datapipes into one
    return DictDatasetIterDataPipe(
        {k: v for k, v in datapipe_dict.items() if k != "config"},
    ).map(combine_to_single_dataset)


def split_dataset_dict_dp(element):
    """
    Wrap each of the data source inputs into a datapipe

    Args:
        element: Dictionary of xarray objects
    """

    element = flatten_nwp_source_dict(element)
    output_dict = {k: IterableWrapper([v]) for k, v in element.items() if k != "config"}
    output_dict = nest_nwp_source_dict(output_dict)

    return output_dict


def pvnet_site_netcdf_datapipe(
    keys: List[str],
    filenames: List[str],
) -> IterDataPipe:
    """
    Load the saved Datapipes from pvnet site, and transform to numpy batch

    Args:
        keys: List of keys to extract from the single NetCDF files
        filenames: List of NetCDF files to load

    Returns:
        Datapipe that transforms the NetCDF files to numpy batch
    """
    logger.info("Constructing pvnet site file pipeline")
    # Load files
    datapipe_dict_dp: IterDataPipe = LoadDictDatasetIterDataPipe(
        filenames=filenames,
        keys=keys,
    ).map(split_dataset_dict_dp)
    datapipe = datapipe_dict_dp.pvnet_site_convert_to_numpy_batch()

    return datapipe


if __name__ == "__main__":
    # Load the saved NetCDF files here
    datapipe = pvnet_site_netcdf_datapipe(
        keys=["nwp", "pv"],
        filenames=["/run/media/jacob/data/train/000000.nc"],
    )
    batch = next(iter(datapipe))
    # print(batch)
    # print(len(batch[BatchKey.pv_solar_azimuth]))
    print(batch[BatchKey.pv_solar_elevation].shape)
