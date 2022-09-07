"""Build MetNet data pipeline"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import (
    OpenConfiguration,
    OpenGSPFromDatabase,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenSatellite,
    OpenTopography,
)

logger = logging.getLogger(__name__)


def build_metnet_dataloader(configuration_filename: Union[Path, str]) -> IterDataPipe:
    """
    Create the MetNet production pipeline using a configuration

    Args:
        configuration_filename: Name of the configuration

    Returns:
        DataPipe ready to be put in a Dataloader for training
    """
    ####################################
    #
    # Equivalent to PP's loading and filtering methods
    #
    #####################################
    # Normalize GSP and PV on whole dataset here
    config_datapipe = OpenConfiguration(configuration_filename)
    # TODO Pass the configuration through all the datapipes instead?
    configuration: Configuration = next(iter(config_datapipe))

    logger.debug("Opening Datasets")
    sat_hrv_datapipe = OpenSatellite(
        zarr_path=configuration.input_data.hrvsatellite.hrvsatellite_zarr_path
    )
    passiv_datapipe = OpenPVFromNetCDF(
        providers=[pv_files.label for pv_files in configuration.input_data.pv.pv_files_groups],
        load_extra_minutes=configuration.input_data.pv.live_load_extra_minutes,
        history_minutes=configuration.input_data.pv.history_minutes,
    )

    nwp_datapipe = OpenNWP(configuration.input_data.nwp.nwp_zarr_path)
    topo_datapipe = OpenTopography(configuration.input_data.topographic.topographic_filename)
    gsp_datapipe = OpenGSPFromDatabase(
        history_minutes=configuration.input_data.gsp.history_minutes,
        interpolate_minutes=configuration.input_data.gsp.live_interpolate_minutes,
        load_extra_minutes=configuration.input_data.gsp.live_load_extra_minutes,
    ).drop_national_gsp()
    logger.debug("Normalize GSP data")
    gsp_datapipe = gsp_datapipe.normalize(
        normalize_fn=lambda x: x / x.capacity_megawatt_power
    ).add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
    )
    logger.debug("Getting locations")
    location_datapipe1, location_datapipe2, location_datapipe3 = gsp_datapipe.location_picker(
        return_all_locations=True
    ).fork(3)
