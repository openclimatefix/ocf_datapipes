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
    open_and_return_datapipes,
    create_t0_and_loc_datapipes,
    slice_datapipes_by_time,
)


from ocf_datapipes.utils.consts import NEW_NWP_MEAN, NEW_NWP_STD, RSS_MEAN, RSS_STD

xarray.set_options(keep_attrs=True)
logger = logging.getLogger("pvnet_datapipe")
logger.setLevel(logging.DEBUG)


def normalize_gsp(x):
    """
    Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.capacity_megawatt_power


def gsp_remove_nans(da_gsp):
    return da_gsp.fillna(0.0)


def pvnet_datapipe(
    configuration: str,
    start_time,
    end_time,    
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
    datapipes_dict = open_and_return_datapipes(
        configuration_filename=configuration_filename,
        use_gsp=  True,
        use_pv=   False,
        use_sat=  True,
        use_hrv=  False,
        use_nwp=  True,
        use_topo= False,
    )
    
    
    datapipes_dict["gsp"] = datapipes_dict["gsp"].remove_northern_gsp()
    
    # Filter to time range
    if (start_time is not None) or (end_time is not None):
        datapipes_dict["gsp"] = datapipes_dict["gsp"].select_train_test_time(start_time, end_time)

    # Now get overlapping time periods
    location_pipe, t0_datapipe = create_t0_and_loc_datapipes(
        datapipes_dict, 
        key_for_t0="gsp",
        shuffle=True,
    )
        
    # And now get time slices
    slice_datapipes_by_time(datapipes_dict, t0_datapipe, split_future=False)
    
    numpy_modalities = []
    
    if "nwp" in datapipes_dict:
        nwp_datapipe = datapipes_dict["nwp"]
        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
            location_pipe_copy,
            roi_height_pixels=conf_nwp.nwp_image_size_pixels_height,
            roi_width_pixels=conf_nwp.nwp_image_size_pixels_width,
            x_dim_name="x_osgb",
            y_dim_name="y_osgb",
            datapipe_name="NWP",
        )
        nwp_datapipe.normalize(mean=NEW_NWP_MEAN, std=NEW_NWP_STD)
        numpy_modalities.append(nwp_datapipe.convert_nwp_to_numpy_batch())
        
    if "sat" in datapipes_dict:
        sat_datapipe = datapipes_dict["sat"]
        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
            location_pipe_copy,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
            datapipe_name="Satellite",
        )
        sat_datapipe = sat_datapipe.normalize(mean=RSS_MEAN, std=RSS_STD)
        numpy_modalities.append(sat_datapipe.convert_satellite_to_numpy_batch())
        
    if "hrv" in datapipes_dict:
        hrv_datapipe = datapipes_dict["hrv"]
        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        hrv_datapipe = hrv_datapipe.select_spatial_slice_pixels(
            location_pipe_copy,
            roi_height_pixels=conf_hrv.hrvsatellite_image_size_pixels_height,
            roi_width_pixels=conf_hrv.hrvsatellite_image_size_pixels_width,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
            datapipe_name="HRVSatellite",
        )
        hrv_datapipe = hrv_datapipe.normalize(mean=RSS_MEAN, std=RSS_STD)
        numpy_modalities.append(hrv_datapipe.convert_satellite_to_numpy_batch(is_hrv=True))
    
    
    gsp_datapipe = datapipes_dict["gsp"]
    gsp_datapipe = gsp_datapipe.select_spatial_slice_meters(
        location_datapipe=location_pipe,
        roi_height_meters=1,
        roi_width_meters=1,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
        dim_name="gsp_id",
        datapipe_name="GSP",
    )
    
    gsp_datapipe = gsp_datapipe.normalize(normalize_fn=normalize_gsp)
    # TODO:
    # - If we fill before normalisation we get NaNs. I think some GSP capacities are NaN
    # - Maybe it is better if the model itself handles NaNs? Progagating them through could allow us
    #   to predict on them, but ignore the predictions in the loss function, rather than pushing
    #   the model to predict 0 for these samples.
    gsp_datapipe = gsp_datapipe.map(gsp_remove_nans)
    numpy_modalities.append(gsp_datapipe.convert_gsp_to_numpy_batch())
    
    
    logger.debug("Combine all the data sources")
    combined_datapipe = (
        MergeNumpyModalities(numpy_modalities)
            .add_sun_position(modality_name="gsp")
    )
    
    return combined_datapipe
