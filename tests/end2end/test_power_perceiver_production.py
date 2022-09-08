import numpy as np
import torch
import torchdata.datapipes as dp
import xarray
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Mapper
from torchdata.datapipes.utils import to_graph

xarray.set_options(keep_attrs=True)

from datetime import timedelta

from ocf_datapipes.batch import MergeNumpyExamplesToBatch, MergeNumpyModalities
from ocf_datapipes.convert import (
    ConvertGSPToNumpyBatch,
    ConvertNWPToNumpyBatch,
    ConvertPVToNumpyBatch,
    ConvertSatelliteToNumpyBatch,
)
from ocf_datapipes.experimental import EnsureNNWPVariables, SetSystemIDsToOne
from ocf_datapipes.select import (
    DropNationalGSP,
    LocationPicker,
    SelectLiveT0Time,
    SelectLiveTimeSlice,
    SelectSpatialSliceMeters,
    SelectSpatialSlicePixels,
    SelectTimeSlice,
)
from ocf_datapipes.transform.numpy import (
    AddSunPosition,
    AddTopographicData,
    AlignGSPto5Min,
    EncodeSpaceTime,
    ExtendTimestepsToFuture,
    SaveT0Time,
)
from ocf_datapipes.transform.xarray import (
    AddT0IdxAndSamplePeriodDuration,
    ConvertSatelliteToInt8,
    ConvertToNWPTargetTime,
    Downsample,
    EnsureNPVSystemsPerExample,
    Normalize,
    ReprojectTopography,
)
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD, SAT_MEAN, SAT_STD, BatchKey


def test_power_perceiver_production(
    sat_hrv_datapipe, passiv_datapipe, topo_datapipe, gsp_datapipe, nwp_datapipe
):
    ####################################
    #
    # Equivalent to PP's loading and filtering methods
    #
    #####################################
    # Normalize GSP and PV on whole dataset here
    pv_datapipe = Normalize(passiv_datapipe, normalize_fn=lambda x: x / x.capacity_watt_power)
    gsp_datapipe, gsp_loc_datapipe = DropNationalGSP(gsp_datapipe).fork(2)
    gsp_datapipe = Normalize(gsp_datapipe, normalize_fn=lambda x: x / x.capacity_megawatt_power)
    topo_datapipe = ReprojectTopography(topo_datapipe)
    sat_datapipe = ConvertSatelliteToInt8(sat_hrv_datapipe)
    sat_datapipe = AddT0IdxAndSamplePeriodDuration(
        sat_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )
    pv_datapipe = AddT0IdxAndSamplePeriodDuration(
        pv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )
    gsp_datapipe, gsp_t0_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe,
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(hours=2),
    ).fork(2)
    nwp_datapipe = AddT0IdxAndSamplePeriodDuration(
        nwp_datapipe, sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2)
    )

    ####################################
    #
    # Equivalent to PP's xr_batch_processors and normal loading/selecting
    #
    #####################################

    location_datapipe1, location_datapipe2, location_datapipe3, location_datapipe4 = LocationPicker(
        gsp_loc_datapipe, return_all_locations=True
    ).fork(
        4
    )  # Its in order then
    pv_datapipe = SelectSpatialSliceMeters(
        pv_datapipe,
        location_datapipe=location_datapipe1,
        roi_width_meters=960_000,
        roi_height_meters=960_000,
    )  # Has to be large as test PV systems aren't in first 20 GSPs it seems
    pv_datapipe, pv_t0_datapipe = EnsureNPVSystemsPerExample(
        pv_datapipe, n_pv_systems_per_example=8
    ).fork(2)
    sat_datapipe, sat_t0_datapipe = SelectSpatialSlicePixels(
        sat_datapipe,
        location_datapipe=location_datapipe2,
        roi_width_pixels=256,
        roi_height_pixels=128,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    ).fork(2)
    nwp_datapipe, nwp_t0_datapipe = SelectSpatialSlicePixels(
        nwp_datapipe,
        location_datapipe=location_datapipe3,
        roi_width_pixels=64,
        roi_height_pixels=64,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
    ).fork(2)
    nwp_datapipe = Downsample(nwp_datapipe, y_coarsen=16, x_coarsen=16)
    nwp_t0_datapipe = SelectLiveT0Time(nwp_t0_datapipe, dim_name="init_time_utc")
    nwp_datapipe = ConvertToNWPTargetTime(
        nwp_datapipe,
        t0_datapipe=nwp_t0_datapipe,
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(hours=2),
        forecast_duration=timedelta(hours=3),
    )
    gsp_t0_datapipe = SelectLiveT0Time(gsp_t0_datapipe)
    gsp_datapipe = SelectLiveTimeSlice(
        gsp_datapipe,
        t0_datapipe=gsp_t0_datapipe,
        history_duration=timedelta(hours=2),
    )
    sat_t0_datapipe = SelectLiveT0Time(sat_t0_datapipe)
    sat_datapipe = SelectLiveTimeSlice(
        sat_datapipe,
        t0_datapipe=sat_t0_datapipe,
        history_duration=timedelta(hours=1),
    )
    passiv_t0_datapipe = SelectLiveT0Time(pv_t0_datapipe)
    pv_datapipe = SelectLiveTimeSlice(
        pv_datapipe,
        t0_datapipe=passiv_t0_datapipe,
        history_duration=timedelta(hours=1),
    )
    gsp_datapipe = SelectSpatialSliceMeters(
        gsp_datapipe,
        location_datapipe=location_datapipe4,
        dim_name="gsp_id",
        roi_width_meters=10,
        roi_height_meters=10,
    )

    sat_datapipe = Normalize(sat_datapipe, mean=SAT_MEAN["HRV"] / 4, std=SAT_STD["HRV"] / 4).map(
        lambda x: x.resample(time_utc="5T").interpolate("linear")
    )  # Interplate to 5 minutes incase its 15 minutes
    nwp_datapipe = Normalize(nwp_datapipe, mean=NWP_MEAN, std=NWP_STD)
    topo_datapipe = Normalize(topo_datapipe, calculate_mean_std_from_example=True)

    ####################################
    #
    # Equivalent to PP's np_batch_processors
    #
    #####################################

    sat_datapipe = ConvertSatelliteToNumpyBatch(sat_datapipe, is_hrv=True)
    sat_datapipe = ExtendTimestepsToFuture(
        sat_datapipe,
        forecast_duration=timedelta(hours=2),
        sample_period_duration=timedelta(minutes=5),
    )
    sat_datapipe = MergeNumpyExamplesToBatch(sat_datapipe, n_examples_per_batch=4)
    pv_datapipe = ConvertPVToNumpyBatch(pv_datapipe)
    pv_datapipe = ExtendTimestepsToFuture(
        pv_datapipe,
        forecast_duration=timedelta(hours=2),
        sample_period_duration=timedelta(minutes=5),
    )
    pv_datapipe = MergeNumpyExamplesToBatch(pv_datapipe, n_examples_per_batch=4)
    gsp_datapipe = ConvertGSPToNumpyBatch(gsp_datapipe)
    gsp_datapipe = ExtendTimestepsToFuture(
        gsp_datapipe,
        forecast_duration=timedelta(hours=8),
        sample_period_duration=timedelta(minutes=30),
    )
    gsp_datapipe = MergeNumpyExamplesToBatch(gsp_datapipe, n_examples_per_batch=4)
    # Don't need to do NWP as it does go into the future
    nwp_datapipe = ConvertNWPToNumpyBatch(nwp_datapipe)
    nwp_datapipe = MergeNumpyExamplesToBatch(nwp_datapipe, n_examples_per_batch=4)
    combined_datapipe = MergeNumpyModalities(
        [gsp_datapipe, pv_datapipe, sat_datapipe, nwp_datapipe]
    )

    combined_datapipe = AlignGSPto5Min(
        combined_datapipe, batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc
    )
    combined_datapipe = EncodeSpaceTime(combined_datapipe)
    combined_datapipe = SaveT0Time(combined_datapipe)
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="hrvsatellite")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="pv")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="gsp")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="gsp_5_min")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="nwp_target_time")
    combined_datapipe = AddTopographicData(combined_datapipe, topo_datapipe)
    combined_datapipe = SetSystemIDsToOne(combined_datapipe)

    batch = next(iter(combined_datapipe))

    assert len(batch[BatchKey.hrvsatellite_time_utc]) == 4
    assert len(batch[BatchKey.hrvsatellite_time_utc][0]) == 37
    assert len(batch[BatchKey.nwp_target_time_utc][0]) == 6
    assert len(batch[BatchKey.nwp_init_time_utc][0]) == 6
    assert len(batch[BatchKey.pv_time_utc][0]) == 37
    assert len(batch[BatchKey.gsp_time_utc][0]) == 21

    assert batch[BatchKey.hrvsatellite_actual].shape == (4, 13, 1, 128, 256)
    assert batch[BatchKey.nwp].shape == (4, 6, 1, 4, 4)
    assert batch[BatchKey.pv].shape == (4, 13, 8)
    assert batch[BatchKey.gsp].shape == (4, 5, 1)
    assert batch[BatchKey.hrvsatellite_surface_height].shape == (4, 128, 256)


def test_power_perceiver_production_functional(
    sat_hrv_datapipe, passiv_datapipe, topo_datapipe, gsp_datapipe, nwp_datapipe
):
    ####################################
    #
    # Equivalent to PP's loading and filtering methods
    #
    #####################################
    # Normalize GSP and PV on whole dataset here

    gsp_datapipe, gsp_loc_datapipe = (
        gsp_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_megawatt_power)
        .drop_national_gsp()
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=2)
        )
        .fork(2)
    )
    (
        location_datapipe1,
        location_datapipe2,
        location_datapipe3,
        location_datapipe4,
    ) = gsp_loc_datapipe.location_picker(return_all_locations=True).fork(4)

    passiv_datapipe, pv_t0_datapipe = (
        passiv_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_watt_power)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
        )
        .select_spatial_slice_meters(
            location_datapipe=location_datapipe1,
            roi_width_meters=960_000,
            roi_height_meters=960_000,
        )
        .ensure_n_pv_systems_per_example(n_pv_systems_per_example=8)
        .fork(2)
    )
    topo_datapipe = topo_datapipe.reproject_topography().normalize(
        calculate_mean_std_from_example=True
    )
    sat_hrv_datapipe, sat_t0_datapipe = (
        sat_hrv_datapipe.convert_satellite_to_int8()
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
        )
        .select_spatial_slice_pixels(
            location_datapipe=location_datapipe2,
            roi_width_pixels=256,
            roi_height_pixels=128,
            y_dim_name="y_geostationary",
            x_dim_name="x_geostationary",
        )
        .fork(2)
    )

    nwp_datapipe, nwp_t0_datapipe = (
        nwp_datapipe.add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2)
        )
        .select_spatial_slice_pixels(
            location_datapipe=location_datapipe3,
            roi_width_pixels=64,
            roi_height_pixels=64,
            y_dim_name="y_osgb",
            x_dim_name="x_osgb",
        )
        .downsample(y_coarsen=16, x_coarsen=16)
        .fork(2)
    )
    gsp_datapipe, gsp_t0_datapipe = gsp_datapipe.fork(2)

    nwp_t0_datapipe = nwp_t0_datapipe.select_live_t0_time(dim_name="init_time_utc")
    gsp_t0_datapipe = gsp_t0_datapipe.select_live_t0_time()
    sat_t0_datapipe = sat_t0_datapipe.select_live_t0_time()
    pv_t0_datapipe = pv_t0_datapipe.select_live_t0_time()

    gsp_datapipe = (
        gsp_datapipe.select_live_time_slice(
            t0_datapipe=gsp_t0_datapipe, history_duration=timedelta(hours=2)
        )
        .select_spatial_slice_meters(
            location_datapipe=location_datapipe4,
            roi_width_meters=10,
            roi_height_meters=10,
            dim_name="gsp_id",
        )
        .convert_gsp_to_numpy_batch()
        .extend_timesteps_to_future(
            forecast_duration=timedelta(hours=8),
            sample_period_duration=timedelta(minutes=30),
        )
        .add_zeroed_future_data(key=BatchKey.gsp, time_key=BatchKey.gsp_time_utc)
        .merge_numpy_examples_to_batch(n_examples_per_batch=4)
    )
    sat_hrv_datapipe = (
        sat_hrv_datapipe.select_live_time_slice(
            t0_datapipe=sat_t0_datapipe,
            history_duration=timedelta(hours=1),
        )
        .normalize(mean=SAT_MEAN["HRV"] / 4, std=SAT_STD["HRV"] / 4)
        .map(
            lambda x: x.resample(time_utc="5T").interpolate("linear")
        )  # Interplate to 5 minutes incase its 15 minutes
        .convert_satellite_to_numpy_batch(is_hrv=True)
        .extend_timesteps_to_future(
            forecast_duration=timedelta(hours=2),
            sample_period_duration=timedelta(minutes=5),
        )
        .merge_numpy_examples_to_batch(n_examples_per_batch=4)
    )
    passiv_datapipe = (
        passiv_datapipe.select_live_time_slice(
            t0_datapipe=pv_t0_datapipe,
            history_duration=timedelta(hours=1),
        )
        .convert_pv_to_numpy_batch()
        .extend_timesteps_to_future(
            forecast_duration=timedelta(hours=2),
            sample_period_duration=timedelta(minutes=5),
        )
        .add_zeroed_future_data(key=BatchKey.pv, time_key=BatchKey.pv_time_utc)
        .merge_numpy_examples_to_batch(n_examples_per_batch=4)
    )
    nwp_datapipe = (
        nwp_datapipe.convert_to_nwp_target_time(
            t0_datapipe=nwp_t0_datapipe,
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(hours=2),
            forecast_duration=timedelta(hours=3),
        )
        .normalize(mean=NWP_MEAN, std=NWP_STD)
        .convert_nwp_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=4)
    )

    ####################################
    #
    # Equivalent to PP's np_batch_processors
    #
    #####################################
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
    )

    batch = next(iter(combined_datapipe))

    assert len(batch[BatchKey.hrvsatellite_time_utc]) == 4
    assert len(batch[BatchKey.hrvsatellite_time_utc][0]) == 37
    assert len(batch[BatchKey.nwp_target_time_utc][0]) == 6
    assert len(batch[BatchKey.nwp_init_time_utc][0]) == 6
    assert len(batch[BatchKey.pv_time_utc][0]) == 37
    assert len(batch[BatchKey.gsp_time_utc][0]) == 21
    assert len(batch[BatchKey.hrvsatellite_solar_azimuth]) == 4
    assert len(batch[BatchKey.hrvsatellite_solar_elevation]) == 4

    assert batch[BatchKey.hrvsatellite_actual].shape == (4, 13, 1, 128, 256)
    assert batch[BatchKey.nwp].shape == (4, 6, 9, 4, 4)
    assert batch[BatchKey.pv].shape == (4, 37, 8)
    assert batch[BatchKey.gsp].shape == (4, 21, 1)
    assert batch[BatchKey.hrvsatellite_surface_height].shape == (4, 128, 256)

    from torch.utils.data import DataLoader

    dl = DataLoader(dataset=combined_datapipe, batch_size=None)
    batch = next(iter(dl))
    assert len(batch[BatchKey.hrvsatellite_time_utc]) == 4
    assert len(batch[BatchKey.hrvsatellite_time_utc][0]) == 37
    assert len(batch[BatchKey.nwp_target_time_utc][0]) == 6
    assert len(batch[BatchKey.nwp_init_time_utc][0]) == 6
    assert len(batch[BatchKey.pv_time_utc][0]) == 37
    assert len(batch[BatchKey.gsp_time_utc][0]) == 21
    assert len(batch[BatchKey.hrvsatellite_solar_azimuth]) == 4
    assert len(batch[BatchKey.hrvsatellite_solar_elevation]) == 4

    assert batch[BatchKey.hrvsatellite_actual].shape == (4, 13, 1, 128, 256)
    assert batch[BatchKey.nwp].shape == (4, 6, 9, 4, 4)
    assert batch[BatchKey.pv].shape == (4, 37, 8)
    assert batch[BatchKey.gsp].shape == (4, 21, 1)
    assert batch[BatchKey.hrvsatellite_surface_height].shape == (4, 128, 256)
