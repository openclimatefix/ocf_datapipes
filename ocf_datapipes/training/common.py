"""Common functionality for datapipes"""
import logging
from datetime import timedelta

from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import (
    OpenConfiguration,
    OpenGSP,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenSatellite,
    OpenTopography,
)

logger = logging.getLogger(__name__)


def open_and_return_datapipes(
    configuration_filename: str,
    use_gsp: bool = True,
    use_nwp: bool = True,
    use_pv: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_topo: bool = True,
) -> dict[IterDataPipe]:
    """
    Open data sources given a configuration and return the list of datapipes

    Args:
        configuration_filename: Path to file to open
        use_nwp: Whether to use NWP data or not
        use_topo: Whether to use topographic data
        use_pv: Whether to open PV data
        use_hrv: Whether to open HRV satellite data
        use_sat: Whether to open non-HRV satellite data

    Returns:
        List of datapipes corresponding to the datapipes to open
    """
    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Check which modalities to use
    if use_nwp:
        use_nwp = True if configuration.input_data.nwp.nwp_zarr_path != "" else False
    if use_pv:
        use_pv = True if configuration.input_data.pv.pv_files_groups[0].pv_filename != "" else False
    if use_sat:
        use_sat = True if configuration.input_data.satellite.satellite_zarr_path != "" else False
    if use_hrv:
        use_hrv = (
            True if configuration.input_data.hrvsatellite.hrvsatellite_zarr_path != "" else False
        )
    if use_topo:
        use_topo = (
            True if configuration.input_data.topographic.topographic_filename != "" else False
        )
    if use_gsp:
        use_gsp = True if configuration.input_data.gsp.gsp_zarr_path != "" else False
    logger.debug(
        f"GSP: {use_gsp} NWP: {use_nwp} Sat: {use_sat},"
        f" HRV: {use_hrv} PV: {use_pv} Topo: {use_topo}"
    )

    used_datapipes = {}

    # Load GSP national data
    if use_gsp:
        logger.debug("Opening GSP Data")
        gsp_datapipe = OpenGSP(
            gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )

        used_datapipes["gsp"] = gsp_datapipe

    # Load NWP data
    if use_nwp:
        logger.debug("Opening NWP Data")
        nwp_datapipe = (
            OpenNWP(configuration.input_data.nwp.nwp_zarr_path)
            .select_channels(configuration.input_data.nwp.nwp_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(hours=1),
                history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            )
        )

        used_datapipes["nwp"] = nwp_datapipe

    if use_sat:
        logger.debug("Opening Satellite Data")
        sat_datapipe = (
            OpenSatellite(configuration.input_data.satellite.satellite_zarr_path)
            .select_channels(configuration.input_data.satellite.satellite_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(
                    minutes=configuration.input_data.satellite.history_minutes
                ),
            )
        )

        used_datapipes["sat"] = sat_datapipe

    if use_hrv:
        logger.debug("Opening HRV Satellite Data")
        sat_hrv_datapipe = OpenSatellite(
            configuration.input_data.hrvsatellite.hrvsatellite_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
        )

        used_datapipes["hrv"] = sat_hrv_datapipe

    if use_pv:
        logger.debug("Opening PV")
        pv_datapipe = OpenPVFromNetCDF(
            pv=configuration.input_data.pv
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        )

        used_datapipes["pv"] = pv_datapipe

    if use_topo:
        logger.debug("Opening Topographic Data")
        topo_datapipe = OpenTopography(configuration.input_data.topographic.topographic_filename)

        used_datapipes["topo"] = topo_datapipe

    return used_datapipes
