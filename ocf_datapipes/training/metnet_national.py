import xarray
from torchdata.datapipes.iter import IterDataPipe
import logging
from typing import Union
from pathlib import Path

xarray.set_options(keep_attrs=True)

from datetime import timedelta

from ocf_datapipes.select import (
    DropGSP,
    LocationPicker,
    SelectLiveT0Time,
    SelectLiveTimeSlice,
    SelectSpatialSliceMeters,
    SelectTimeSlice,
)

from ocf_datapipes.transform.xarray import (
    AddT0IdxAndSamplePeriodDuration,
    ConvertSatelliteToInt8,
    ConvertToNWPTargetTime,
    CreatePVImage,
    Downsample,
    EnsureNPVSystemsPerExample,
    Normalize,
    PreProcessMetNet,
    ReprojectTopography,
)

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import (
    OpenConfiguration,
    OpenGSPFromDatabase,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenSatellite,
    OpenTopography,
    OpenGSPNational
)

from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD, SAT_MEAN, SAT_STD, BatchKey

logger = logging.getLogger("metnet_datapipe")

def metnet_national_datapipe(configuration_filename: Union[Path, str]) -> IterDataPipe:
    """
        Make GSP national data pipe

        Currently only has GSP and NWP's in them

        Args:
            configuration_filename: the configruation filename for the pipe

        Returns: datapipe
        """

    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Load GSP national data
    logger.debug("Opening GSP Data")
    gsp_datapipe = OpenGSPNational(
        gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
    )

    # Load NWP data
    logger.debug("Opening NWP Data")
    nwp_datapipe = OpenNWP(configuration.input_data.nwp.nwp_zarr_path)

    logger.debug("Add t0 idx and normalize")
    gsp_datapipe, gsp_time_periods_datapipe, gsp_t0_datapipe = (
        gsp_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_megawatt_power)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )
        .fork(3)
    )

    nwp_datapipe, nwp_time_periods_datapipe = nwp_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
    ).fork(2)


    # Satellite
    logger.debug("Opening Satellite Data")
    sat_datapipe = OpenSatellite(configuration.input_data.satellite.satellite_zarr_path)
    sat_hrv_datapipe = OpenSatellite(configuration.input_data.hrvsatellite.hrvsatellite_zarr_path)

    sat_datapipe, sat_time_periods_datapipe = sat_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
    ).fork(2)
    sat_hrv_datapipe, sat_hrv_time_periods_datapipe = sat_hrv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.hrvsatellite.history_minutes),
    ).fork(2)

    # PV
    logger.debug("Opening Datasets")
    pv_datapipe, pv_location_datapipe = OpenPVFromNetCDF(
        pv_power_filename=configuration.input_data.pv.pv_files_groups[0].pv_filename,
        pv_metadata_filename=configuration.input_data.pv.pv_files_groups[0].pv_metadata_filename,
    ).fork(2)

    logger.debug("Add t0 idx")
    (
        pv_datapipe,
        pv_time_periods_datapipe,
    ) = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    ).fork(
        2
    )


    # get time periods
    # get contiguous time periods
    logger.debug("Getting contiguous time periods")
    gsp_time_periods_datapipe = gsp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
    )
    nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        time_dim="init_time_utc",
    )
    pv_time_periods_datapipe = pv_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=1),
    )
    sat_hrv_time_periods_datapipe = sat_hrv_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.hrvsatellite.history_minutes),
        forecast_duration=timedelta(minutes=1),
    )
    sat_time_periods_datapipe = sat_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
        forecast_duration=timedelta(minutes=1),
    )

    # find joint overlapping timer periods
    logger.debug("Getting joint time periods")
    overlapping_datapipe = gsp_time_periods_datapipe.select_overlapping_time_slice(
        secondary_datapipes=[nwp_time_periods_datapipe, pv_time_periods_datapipe, sat_hrv_time_periods_datapipe, sat_time_periods_datapipe],
    )
    gsp_time_periods, nwp_time_periods, sat_time_periods, sat_hrv_time_periods, pv_time_periods = overlapping_datapipe.fork(5, buffer_size=100)

    # select time periods
    gsp_t0_datapipe = gsp_t0_datapipe.select_time_periods(time_periods=gsp_time_periods)

    # select t0 periods
    logger.debug("Select t0 joint")
    gsp_t0_datapipe, nwp_t0_datapipe, sat_t0_datapipe, sat_hrv_t0_datapipe, pv_t0_datapipe = gsp_t0_datapipe.select_t0_time().fork(5)

    # take pv time slices
    logger.debug("Take GSP time slices")
    gsp_datapipe, gsp_loc_datapipe = gsp_datapipe.select_time_slice(
            t0_datapipe=gsp_t0_datapipe,
            history_duration=timedelta(minutes=0),
            forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
            sample_period_duration=timedelta(minutes=30),
        ).fork(2)

    # take nwp time slices
    logger.debug("Take NWP time slices")
    nwp_datapipe = nwp_datapipe.convert_to_nwp_target_time(
            t0_datapipe=nwp_t0_datapipe,
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        ).normalize(mean=NWP_MEAN, std=NWP_STD)

    # take sat time slices
    sat_datapipe = sat_datapipe.select_time_slice(
        t0_datapipe=sat_t0_datapipe,
        history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
        forecast_duration=timedelta(minutes=0),
        sample_period_duration=timedelta(minutes=5),
    ).normalize(mean=SAT_MEAN, std=SAT_STD)
    sat_hrv_datapipe, sat_pv_image_datapipe = sat_hrv_datapipe.select_time_slice(
        t0_datapipe=sat_hrv_t0_datapipe,
        history_duration=timedelta(minutes=configuration.input_data.hrvsatellite.history_minutes),
        forecast_duration=timedelta(minutes=0),
        sample_period_duration=timedelta(minutes=5),
    ).normalize(mean=SAT_MEAN["HRV"], std=SAT_STD["HRV"]).fork(2)

    # take pv time slices
    pv_datapipe = pv_datapipe.select_time_slice(
        t0_datapipe=pv_t0_datapipe,
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=0),
        sample_period_duration=timedelta(minutes=5),
    ).create_pv_image(sat_pv_image_datapipe)

    location_datapipe = LocationPicker(gsp_loc_datapipe, return_all_locations=True)

    # Now combine in the MetNet format
    combined_datapipe = PreProcessMetNet(
        [
            sat_hrv_datapipe,
            sat_datapipe,
            nwp_datapipe,
            pv_datapipe,
        ],
        location_datapipe=location_datapipe,
        center_width=500_000,
        center_height=1_000_000,
        context_height=10_000_000,
        context_width=10_000_000,
        output_width_pixels=256,
        output_height_pixels=512,
    )

    return combined_datapipe.zip(gsp_datapipe) # Makes (Inputs, Label) tuples
