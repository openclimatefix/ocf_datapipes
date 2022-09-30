"""Wrapper for Power Perceiver Production Data Pipeline"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import (
    OpenConfiguration,
    OpenGSPFromDatabase,
    OpenNWP,
    OpenPVFromDB,
    OpenSatellite,
    OpenTopography,
)
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD, SAT_MEAN, SAT_STD, BatchKey

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)


def power_perceiver_production_datapipe(configuration_filename: Union[Path, str]) -> IterDataPipe:
    """
    Create the Power Perceiver production pipeline using a configuration

    Args:
        configuration_filename: Name of the configuration

    Returns:
        DataPipe ready to be put in a Dataloader for production
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
    passiv_datapipe = OpenPVFromDB(
        providers=[pv_files.label for pv_files in configuration.input_data.pv.pv_files_groups],
        load_extra_minutes=configuration.input_data.pv.live_load_extra_minutes,
        history_minutes=configuration.input_data.pv.history_minutes,
    )

    nwp_datapipe = OpenNWP(configuration.input_data.nwp.nwp_zarr_path)
    topo_datapipe = OpenTopography(configuration.input_data.topographic.topographic_filename)
    gsp_datapipe, gso_loc_datapipe = (
        OpenGSPFromDatabase(
            history_minutes=configuration.input_data.gsp.history_minutes,
            interpolate_minutes=configuration.input_data.gsp.live_interpolate_minutes,
            load_extra_minutes=configuration.input_data.gsp.live_load_extra_minutes,
        )
        .drop_gsp()
        .fork(2)
    )
    logger.debug("Normalize GSP data")
    gsp_datapipe, gsp_t0_datapipe = (
        gsp_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_megawatt_power)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )
        .fork(2)
    )
    logger.debug("Getting locations")
    (
        location_datapipe1,
        location_datapipe2,
        location_datapipe3,
        location_datapipe4,
    ) = gso_loc_datapipe.location_picker(return_all_locations=True).fork(4)

    logger.debug("Got locations")

    logger.debug("Making PV space slice")
    passiv_datapipe, pv_t0_datapipe = (
        passiv_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_watt_power)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        )
        .select_spatial_slice_meters(
            location_datapipe=location_datapipe1,
            roi_width_meters=configuration.input_data.pv.pv_image_size_meters_width,
            roi_height_meters=configuration.input_data.pv.pv_image_size_meters_height,
        )
        .ensure_n_pv_systems_per_example(
            n_pv_systems_per_example=configuration.input_data.pv.n_pv_systems_per_example
        )
        .fork(2)
    )
    topo_datapipe = topo_datapipe.reproject_topography().normalize(
        calculate_mean_std_from_example=True
    )
    sat_hrv_datapipe, sat_t0_datapipe = (
        sat_hrv_datapipe.convert_satellite_to_int8()
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
        )
        .select_spatial_slice_pixels(
            location_datapipe=location_datapipe2,
            roi_width_pixels=configuration.input_data.hrvsatellite.hrvsatellite_image_size_pixels_width,  # noqa
            roi_height_pixels=configuration.input_data.hrvsatellite.hrvsatellite_image_size_pixels_height,  # noqa
            y_dim_name="y_geostationary",
            x_dim_name="x_geostationary",
        )
        .fork(2)
    )

    logger.debug("Making NWP space slice")
    nwp_datapipe, nwp_t0_datapipe = (
        nwp_datapipe.add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        )
        .select_spatial_slice_pixels(
            location_datapipe=location_datapipe3,
            roi_width_pixels=configuration.input_data.nwp.nwp_image_size_pixels_width
            * 16,  # TODO What to do here with configurations and such
            roi_height_pixels=configuration.input_data.nwp.nwp_image_size_pixels_height * 16,
            y_dim_name="y_osgb",
            x_dim_name="x_osgb",
        )
        .downsample(y_coarsen=16, x_coarsen=16)
        .fork(2)
    )

    nwp_t0_datapipe = nwp_t0_datapipe.select_live_t0_time(dim_name="init_time_utc")
    gsp_t0_datapipe = gsp_t0_datapipe.select_live_t0_time()
    sat_t0_datapipe = sat_t0_datapipe.select_live_t0_time()
    pv_t0_datapipe = pv_t0_datapipe.select_live_t0_time()

    logger.debug("Making GSP Time slices")
    gsp_datapipe = (
        gsp_datapipe.select_live_time_slice(
            t0_datapipe=gsp_t0_datapipe,
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )
        .select_spatial_slice_meters(
            location_datapipe=location_datapipe4,
            roi_width_meters=10,
            roi_height_meters=10,
            dim_name="gsp_id",
        )
        .convert_gsp_to_numpy_batch()
        .extend_timesteps_to_future(
            forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
            sample_period_duration=timedelta(minutes=30),
        )
        .add_zeroed_future_data(key=BatchKey.gsp, time_key=BatchKey.gsp_time_utc)
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )
    logger.debug("Making Sat Time slices")
    sat_hrv_datapipe = (
        sat_hrv_datapipe.select_live_time_slice(
            t0_datapipe=sat_t0_datapipe,
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
        )
        .normalize(mean=SAT_MEAN["HRV"] / 4, std=SAT_STD["HRV"] / 4)
        .map(
            lambda x: x.resample(time_utc="5T").interpolate("linear")
        )  # Interplate to 5 minutes incase its 15 minutes
        .convert_satellite_to_numpy_batch(is_hrv=True)
        .extend_timesteps_to_future(
            forecast_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.forecast_minutes
            ),
            sample_period_duration=timedelta(minutes=5),
        )
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )
    passiv_datapipe = (
        passiv_datapipe.select_live_time_slice(
            t0_datapipe=pv_t0_datapipe,
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        )
        .convert_pv_to_numpy_batch()
        .extend_timesteps_to_future(
            forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
            sample_period_duration=timedelta(minutes=5),
        )
        .add_zeroed_future_data(key=BatchKey.pv, time_key=BatchKey.pv_time_utc)
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )
    nwp_datapipe = (
        nwp_datapipe.convert_to_nwp_target_time(
            t0_datapipe=nwp_t0_datapipe,
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        )
        .normalize(mean=NWP_MEAN, std=NWP_STD)
        .convert_nwp_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )

    ####################################
    #
    # Equivalent to PP's np_batch_processors
    #
    #####################################
    logger.debug("Combine all the data sources")
    combined_datapipe = (
        MergeNumpyModalities([gsp_datapipe, passiv_datapipe, sat_hrv_datapipe, nwp_datapipe])
        .align_gsp_to_5_min(batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc)
        .encode_space_time()
        .save_t0_time()
        .add_sun_position(modality_name="hrvsatellite")
        .add_sun_position(modality_name="pv")
        .add_sun_position(modality_name="gsp")
        .add_sun_position(modality_name="gsp_5_min")
        .add_sun_position(modality_name="nwp_target_time")
        .add_topographic_data(topo_datapipe)
        .set_system_ids_to_one()
        .ensure_n_nwp_variables(num_variables=9)
        .change_float32()
    )

    return combined_datapipe
