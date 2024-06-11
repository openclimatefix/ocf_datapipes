"""Create the training/validation datapipe for training the PVNet Model"""

import logging
from datetime import datetime
from typing import Optional

import xarray as xr
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.batch import MergeNumpyModalities, MergeNWPNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.training.common import (
    _get_datapipes_dict,
    check_nans_in_satellite_data,
    concat_xr_time_utc,
    construct_loctime_pipelines,
    fill_nans_in_arrays,
    fill_nans_in_pv,
    normalize_gsp,
    normalize_pv,
    slice_datapipes_by_time,
)
from ocf_datapipes.utils.consts import (
    NWP_MEANS,
    NWP_STDS,
    RSS_MEAN,
    RSS_STD,
)

xr.set_options(keep_attrs=True)
logger = logging.getLogger("pvnet_datapipe")


def slice_datapipes_by_space(
    datapipes_dict: dict,
    location_pipe: IterDataPipe,
    configuration: Configuration,
) -> None:
    """Modifies a dictionary of datapipes in-place to yield slices around a given location

    Args:
        datapipes_dict: Dictionary of used datapipes and t0 ones
        location_pipe: Datapipe which yields location for sample
        configuration: Configuration object.
    """

    # Unpack for convenience
    conf_sat = configuration.input_data.satellite
    conf_nwp = configuration.input_data.nwp

    if "nwp" in datapipes_dict:
        for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():
            location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
            datapipes_dict["nwp"][nwp_key] = nwp_datapipe.select_spatial_slice_pixels(
                location_pipe_copy,
                roi_height_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_height,
                roi_width_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_width,
            )

    if "sat" in datapipes_dict:
        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        datapipes_dict["sat"] = datapipes_dict["sat"].select_spatial_slice_pixels(
            location_pipe_copy,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
        )

    # GSP always assumed to be in data
    datapipes_dict["gsp"] = datapipes_dict["gsp"].select_spatial_slice_meters(
        location_datapipe=location_pipe,
        roi_height_meters=1,
        roi_width_meters=1,
        dim_name="gsp_id",
    )

    return


def process_and_combine_datapipes(datapipes_dict, configuration, check_satellite_no_nans=False):
    """Normalize and convert data to numpy arrays"""

    # Unpack for convenience
    conf_nwp = configuration.input_data.nwp

    numpy_modalities = []

    # Normalise the inputs and convert to numpy format
    if "nwp" in datapipes_dict:
        nwp_numpy_modalities = dict()

        for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():
            nwp_datapipe = nwp_datapipe.normalize(
                mean=NWP_MEANS[conf_nwp[nwp_key].nwp_provider],
                std=NWP_STDS[conf_nwp[nwp_key].nwp_provider],
            )
            nwp_numpy_modalities[nwp_key] = nwp_datapipe.convert_nwp_to_numpy_batch()

        # Combine the NWPs into NumpyBatch
        nwp_numpy_modalities = MergeNWPNumpyModalities(nwp_numpy_modalities)
        numpy_modalities.append(nwp_numpy_modalities)

    if "sat" in datapipes_dict:
        sat_datapipe = datapipes_dict["sat"]
        sat_datapipe = sat_datapipe.normalize(mean=RSS_MEAN, std=RSS_STD)
        # Check for large amount of zeros
        sat_datapipe = sat_datapipe.check_value_equal_to_fraction(value=0.0, fraction=0.9)
        numpy_modalities.append(sat_datapipe.convert_satellite_to_numpy_batch())

    if "pv" in datapipes_dict:
        # Recombine PV arrays - see function doc for further explanation
        pv_datapipe = (
            datapipes_dict["pv"].zip_ocf(datapipes_dict["pv_future"]).map(concat_xr_time_utc)
        )
        pv_datapipe = pv_datapipe.normalize(normalize_fn=normalize_pv)
        pv_datapipe = pv_datapipe.map(fill_nans_in_pv)

        numpy_modalities.append(pv_datapipe.convert_pv_to_numpy_batch())

    # GSP always assumed to be in data
    gsp_future_datapipe = datapipes_dict["gsp_future"]
    gsp_datapipe = datapipes_dict["gsp"]

    # Recombine GSP arrays - see function doc for further explanation
    gsp_datapipe = gsp_datapipe.zip_ocf(gsp_future_datapipe).map(concat_xr_time_utc)
    gsp_datapipe = gsp_datapipe.normalize(normalize_fn=normalize_gsp)

    numpy_modalities.append(gsp_datapipe.convert_gsp_to_numpy_batch())

    logger.debug("Combine all the data sources")
    combined_datapipe = MergeNumpyModalities(numpy_modalities).add_sun_position(modality_name="gsp")

    if check_satellite_no_nans:
        # in production we don't want any nans in the satellite data
        combined_datapipe = combined_datapipe.map(check_nans_in_satellite_data)

    combined_datapipe = combined_datapipe.map(fill_nans_in_arrays)

    return combined_datapipe


def construct_sliced_data_pipeline(
    config_filename: str,
    location_pipe: IterDataPipe,
    t0_datapipe: IterDataPipe,
    production: bool = False,
    check_satellite_no_nans: bool = False,
) -> IterDataPipe:
    """Constructs data pipeline for the input data config file.

    This yields samples from the location and time datapipes.

    Args:
        config_filename: Path to config file.
        location_pipe: Datapipe yielding locations.
        t0_datapipe: Datapipe yielding times.
        production: Whether constucting pipeline for production inference.
        check_satellite_no_nans: Whether to check that satellite data has no nans.
    """

    datapipes_dict = _get_datapipes_dict(
        config_filename,
        production=production,
    )

    configuration = datapipes_dict.pop("config")

    # Slice all of the datasets by space - this is an in-place operation
    slice_datapipes_by_space(datapipes_dict, location_pipe, configuration)

    # Slice all of the datasets by time - this is an in-place operation
    slice_datapipes_by_time(datapipes_dict, t0_datapipe, configuration, production)

    # Normalise, and combine the data sources into NumpyBatches
    combined_datapipe = process_and_combine_datapipes(
        datapipes_dict, configuration, check_satellite_no_nans
    )

    return combined_datapipe


def pvnet_datapipe(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> IterDataPipe:
    """
    Construct pvnet pipeline for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time at which a sample can be selected.
        end_time: Maximum time at which a sample can be selected.
    """
    logger.info("Constructing pvnet pipeline")

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
    datapipe = construct_sliced_data_pipeline(
        config_filename,
        location_pipe,
        t0_datapipe,
    )

    return datapipe
