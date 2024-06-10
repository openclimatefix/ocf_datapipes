"""Data pipeline for loading GSP, PV, Satellite and NWP"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray as xr
from torch.utils.data.datapipes.datapipe import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities, MergeNWPNumpyModalities
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite
from ocf_datapipes.training.common import normalize_gsp, normalize_pv
from ocf_datapipes.utils.consts import NWP_MEANS, NWP_STDS, RSS_MEAN, RSS_STD

logger = logging.getLogger(__name__)
xr.set_options(keep_attrs=True)

# should scale with batch_size #TODO
BUFFER_SIZE = 100


def gsp_pv_nwp_satellite_data_pipeline(configuration: Union[Path, str]) -> IterDataPipe:
    """
    Make data pipe with GSP, PV, NWP and Satellite

    The locations are chosen by sampling from GSPs

    Args:
        configuration: the configuration filename for the pipe, can also be the actual configuration

    Returns: datapipe
    """

    # ----- LOAD AND NORMALIZE DATA

    # Load configuration
    configuration: Configuration = load_yaml_configuration(configuration)

    # Load and normalize GSP national data
    gsp_datapipe = OpenGSP(
        gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
    ).normalize(normalize_fn=normalize_gsp)

    # Load and normalize PV data, and fill night-time NaNs
    pv_datapipe = (
        OpenPVFromNetCDF(pv=configuration.input_data.pv)
        .pv_fill_night_nans()
        .normalize(normalize_fn=normalize_pv)
    )

    # Load and noralize satellite data
    satellite_datapipe = OpenSatellite(
        zarr_path=configuration.input_data.satellite.satellite_zarr_path
    ).normalize(mean=RSS_MEAN, std=RSS_STD)

    # Load and normalize NWP data - There may be multiple NWP sources
    nwp_datapipe_dict = {}
    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_datapipe_dict[nwp_source] = OpenNWP(
            nwp_conf.nwp_zarr_path, provider=nwp_conf.nwp_provider
        ).normalize(mean=NWP_MEANS[nwp_conf.nwp_provider], std=NWP_STDS[nwp_conf.nwp_provider])

    # ----- ADD t0 IDX

    # Add t0 idx to GSP
    gsp_datapipe = gsp_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.gsp.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
    )

    # Add t0 idx to PV
    pv_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    )

    # Add t0 idx to satellite
    satellite_datapipe = satellite_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.satellite.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
    )

    # Add t0 idx to NWP and fork for time periods overlap
    nwp_time_periods_datapipes = {}
    for nwp_source in nwp_datapipe_dict.keys():
        nwp_datapipe_dict[nwp_source], nwp_time_periods_datapipes[nwp_source] = (
            nwp_datapipe_dict[nwp_source]
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(hours=3),
                history_duration=timedelta(
                    minutes=configuration.input_data.nwp[nwp_source].history_minutes
                ),
            )
            .fork(2)
        )

    # ----- CORE DATASET FORK

    # Fork the GSP data for different uses
    (
        gsp_datapipe,
        gsp_time_periods_datapipe,
        gsp_valid_times_datapipe,
        gsp_location_datapipe,
    ) = gsp_datapipe.fork(4)

    # ----- LOAD AND NORMALIZE DATA

    # Pick locations
    location_datapipe = gsp_location_datapipe.pick_locations()

    # Take PV space slice
    loc_dp, location_datapipe = location_datapipe.fork(2, buffer_size=BUFFER_SIZE)

    pv_datapipe = pv_datapipe.select_spatial_slice_meters(
        location_datapipe=loc_dp,
        roi_height_meters=configuration.input_data.pv.pv_image_size_meters_height,
        roi_width_meters=configuration.input_data.pv.pv_image_size_meters_width,
        dim_name="pv_system_id",
    )

    # Take satellite space slice
    loc_dp, location_datapipe = location_datapipe.fork(2, buffer_size=BUFFER_SIZE)

    satellite_datapipe = satellite_datapipe.select_spatial_slice_pixels(
        location_datapipe=loc_dp,
        roi_height_pixels=configuration.input_data.satellite.satellite_image_size_pixels_height,
        roi_width_pixels=configuration.input_data.satellite.satellite_image_size_pixels_width,
    )

    # Take NWP space slice
    nwp_numpy_modalities = dict()

    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        loc_dp, location_datapipe = location_datapipe.fork(2, buffer_size=BUFFER_SIZE)

        nwp_datapipe_dict[nwp_source] = nwp_datapipe_dict[nwp_source].select_spatial_slice_pixels(
            loc_dp,
            roi_height_pixels=nwp_conf.nwp_image_size_pixels_height,
            roi_width_pixels=nwp_conf.nwp_image_size_pixels_width,
        )

    # Take GSP space slice
    gsp_datapipe = gsp_datapipe.select_spatial_slice_meters(
        location_datapipe=location_datapipe,
        roi_height_meters=configuration.input_data.gsp.gsp_image_size_pixels_height,
        roi_width_meters=configuration.input_data.gsp.gsp_image_size_pixels_width,
        dim_name="gsp_id",
    )

    # ----- SELECT TIME SLICES

    # GSP get contiguous time periods
    gsp_time_periods_datapipe = gsp_time_periods_datapipe.find_contiguous_t0_time_periods(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
    )

    # PV get contiguous time periods
    pv_datapipe, pv_time_periods_datapipe = pv_datapipe.fork(2)

    pv_time_periods_datapipe = pv_time_periods_datapipe.find_contiguous_t0_time_periods(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    )

    # Satellite get contiguous time periods
    satellite_datapipe, satellite_time_periods_datapipe = satellite_datapipe.fork(2)

    satellite_time_periods_datapipe = (
        satellite_time_periods_datapipe.find_contiguous_t0_time_periods(
            sample_period_duration=timedelta(
                minutes=configuration.input_data.satellite.time_resolution_minutes
            ),
            history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
            forecast_duration=timedelta(
                minutes=configuration.input_data.satellite.forecast_minutes
            ),
        )
    )

    # NWP get contiguous time periods
    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_time_periods_datapipes[nwp_source] = nwp_time_periods_datapipes[
            nwp_source
        ].find_contiguous_t0_time_periods(
            sample_period_duration=timedelta(hours=3),
            history_duration=timedelta(minutes=nwp_conf.history_minutes),
            forecast_duration=timedelta(minutes=nwp_conf.forecast_minutes),
            time_dim="init_time_utc",
        )

    # Find joint overlapping time periods
    overlapping_datapipe = gsp_time_periods_datapipe.filter_to_overlapping_time_periods(
        secondary_datapipes=[
            pv_time_periods_datapipe,
            satellite_time_periods_datapipe,
            *nwp_time_periods_datapipes.values(),
        ],
    )

    # Filter to times valid for all data sources
    valid_periods_datapipe = gsp_valid_times_datapipe.filter_time_periods(
        time_periods=overlapping_datapipe
    )

    # Select t0 times
    t0_datapipe = valid_periods_datapipe.pick_t0_times()

    # Take NWP time slices and convert to NumpyBatch
    nwp_numpy_modalities = dict()

    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_t0_datapipe, t0_datapipe = t0_datapipe.fork(2)

        nwp_numpy_modalities[nwp_source] = (
            nwp_datapipe_dict[nwp_source]
            .select_time_slice_nwp(
                t0_datapipe=nwp_t0_datapipe,
                sample_period_duration=timedelta(hours=3),
                history_duration=timedelta(minutes=nwp_conf.history_minutes),
                forecast_duration=timedelta(minutes=nwp_conf.forecast_minutes),
            )
            .convert_nwp_to_numpy_batch()
        )

    # Combine the NWPs into NumpyBatch
    nwps_numpy_datapipe = MergeNWPNumpyModalities(nwp_numpy_modalities)

    # Take PV time slices and convert to NumpyBatch
    pv_t0_datapipe, t0_datapipe = t0_datapipe.fork(2)

    pv_numpy_datapipe = pv_datapipe.select_time_slice(
        t0_datapipe=pv_t0_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    ).convert_pv_to_numpy_batch()

    # Take satellite time slices and convert to NumpyBatch
    satellite_t0_datapipe, t0_datapipe = t0_datapipe.fork(2)

    satellite_numpy_datapipe = satellite_datapipe.select_time_slice(
        t0_datapipe=satellite_t0_datapipe,
        sample_period_duration=timedelta(
            minutes=configuration.input_data.satellite.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.satellite.forecast_minutes),
    ).convert_satellite_to_numpy_batch()

    # Take GSP time slices and convert to NumpyBatch
    gsp_numpy_datapipe = gsp_datapipe.select_time_slice(
        t0_datapipe=t0_datapipe,
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
        sample_period_duration=timedelta(minutes=30),
    ).convert_gsp_to_numpy_batch()

    # Join datapipes together
    combined_datapipe = MergeNumpyModalities(
        [gsp_numpy_datapipe, pv_numpy_datapipe, satellite_numpy_datapipe, nwps_numpy_datapipe]
    )

    # Now batch the data with batch size 4
    combined_datapipe = combined_datapipe.batch(4).merge_numpy_batch()

    return combined_datapipe
