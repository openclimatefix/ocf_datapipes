"""Create the training/validation datapipe for training the PVNet Model"""
import datetime
import logging
from pathlib import Path
from typing import Union
import fsspec
from pyaml_env import parse_config

import xarray
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.convert import ConvertPVToNumpy
from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.training.common import (
    add_selected_time_slices_from_datapipes,
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
)
from ocf_datapipes.utils.consts import NEW_NWP_MEAN, NEW_NWP_STD, RSS_MEAN, RSS_STD

xarray.set_options(keep_attrs=True)
logger = logging.getLogger("pvnet_datapipe")
logger.setLevel(logging.DEBUG)


def normalize_gsp(x):  # So it can be pickled
    """
    Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.capacity_megawatt_power


def pvnet_datapipe(
    configuration: str,
    start_time,
    end_time,
    add_sun=True,
    
) -> IterDataPipe:
    """
    Make data pipe with GSP, NWP and Satellite

    Args:
        configuration: the configuration filename for the pipe, can also be the actual configuration

    Returns: datapipe
    """

    # load configuration
    configuration_filename = configuration
    with fsspec.open(configuration, mode="r") as stream:
        configuration = parse_config(data=stream)
    configuration = Configuration(**configuration)
        
    conf_sat = configuration.input_data.satellite
    conf_hrv = configuration.input_data.hrvsatellite
    conf_nwp = configuration.input_data.nwp

    # load datasets
    used_datapipes = open_and_return_datapipes(
        configuration_filename=configuration_filename,
        use_gsp=  True,
        use_pv=   False,
        use_sat=  True,
        use_hrv=  False,
        use_nwp=  True,
        use_topo= False,
    )
    # These now only return one-time-yielding iters
    
    print(f"\n\n\n{used_datapipes.keys()}\n\n\n")
    
    used_datapipes["gsp"] = used_datapipes["gsp"].remove_northern_gsp()
    
    # Filter to time range
    used_datapipes["gsp"] = used_datapipes["gsp"].select_train_test_time(start_time, end_time)

    # Now get overlapping time periods
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes, key_for_t0="gsp")

    # And now get time slices
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes, split_future=False)

    # Now do the extra processing
    gsp_datapipe, location_pipe = used_datapipes["gsp"].fork(2)
    
    gsp_datapipe = gsp_datapipe.normalize(normalize_fn=normalize_gsp)
    numpy_modalities = [gsp_datapipe.convert_gsp_to_numpy_batch()]
    
    location_pipe = location_pipe.location_picker()
    
    
    if "nwp" in used_datapipes.keys():
        logger.debug("Take NWP time slices")
        nwp_image_loc_datapipe, location_pipe = location_pipe.fork(2)
        nwp_datapipe = used_datapipes["nwp"].normalize(mean=NEW_NWP_MEAN, std=NEW_NWP_STD)
        # context_size is the largest it would need
        nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
            nwp_image_loc_datapipe,
            roi_height_pixels=conf_nwp.nwp_image_size_pixels_height,
            roi_width_pixels=conf_nwp.nwp_image_size_pixels_width,
            x_dim_name="x_osgb",
            y_dim_name="y_osgb",
            datapipe_name="NWP",
        )
        
        numpy_modalities.append(nwp_datapipe.convert_nwp_to_numpy_batch())
        
    if "sat" in used_datapipes.keys():
        logger.debug("Take Satellite time slices")
        sat_image_loc_datapipe, location_pipe = location_pipe.fork(2)
        sat_datapipe = used_datapipes["sat"].normalize(mean=RSS_MEAN, std=RSS_STD)
        sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
            sat_image_loc_datapipe,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
            datapipe_name="Satellite",
        )
        numpy_modalities.append(sat_datapipe.convert_satellite_to_numpy_batch())
        
    if "hrv" in used_datapipes.keys():
        logger.debug("Take HRV Satellite time slices")
        hrv_image_loc_datapipe, location_pipe = location_pipe.fork(2)
        hrv_datapipe = used_datapipes["hrv"].normalize(mean=RSS_MEAN, std=RSS_STD)
        hrv_datapipe = hrv_datapipe.select_spatial_slice_pixels(
            hrv_image_loc_datapipe,
            roi_height_pixels=conf_hrv.hrvsatellite_image_size_pixels_height,
            roi_width_pixels=conf_hrv.hrvsatellite_image_size_pixels_width,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
            datapipe_name="HRVSatellite",
        )
        numpy_modalities.append(hrv_datapipe.convert_satellite_to_numpy_batch(is_hrv=True))
    
    logger.debug("Combine all the data sources")
    combined_datapipe = (
        MergeNumpyModalities(numpy_modalities)
        # .encode_space_time()
        #.add_sun_position(modality_name="gsp")
    )
    
    if add_sun:
        combined_datapipe = combined_datapipe.add_sun_position(modality_name="gsp")

    return combined_datapipe
