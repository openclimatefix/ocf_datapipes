import numpy as np
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
from ocf_datapipes.production.power_perceiver import GSPIterator
from ocf_datapipes.select import (
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


def test_power_perceiver_production(sat_hrv_dp, passiv_dp, topo_dp, gsp_dp, nwp_dp):
    ####################################
    #
    # Equivalent to PP's loading and filtering methods
    #
    #####################################
    # Normalize GSP and PV on whole dataset here
    pv_dp = Normalize(passiv_dp, normalize_fn=lambda x: x / x.capacity_wp)
    gsp_dp = Normalize(gsp_dp, normalize_fn=lambda x: x / x.capacity_mwp)
    topo_dp = ReprojectTopography(topo_dp)
    sat_dp = ConvertSatelliteToInt8(sat_hrv_dp)
    sat_dp = AddT0IdxAndSamplePeriodDuration(
        sat_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    pv_dp = AddT0IdxAndSamplePeriodDuration(
        pv_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    gsp_dp = AddT0IdxAndSamplePeriodDuration(
        gsp_dp, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=2)
    )
    nwp_dp = AddT0IdxAndSamplePeriodDuration(
        nwp_dp, sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2)
    )

    ####################################
    #
    # Equivalent to PP's xr_batch_processors and normal loading/selecting
    #
    #####################################

    location_dp1, location_dp2, location_dp3 = LocationPicker(
        gsp_dp, return_all_locations=True
    ).fork(
        3
    )  # Its in order then
    pv_dp = SelectSpatialSliceMeters(
        pv_dp, location_datapipe=location_dp1, roi_width_meters=960_000, roi_height_meters=960_000
    )  # Has to be large as test PV systems aren't in first 20 GSPs it seems
    pv_dp, pv_t0_dp = EnsureNPVSystemsPerExample(pv_dp, n_pv_systems_per_example=8).fork(2)
    sat_dp, sat_t0_dp = SelectSpatialSlicePixels(
        sat_dp,
        location_datapipe=location_dp2,
        roi_width_pixels=256,
        roi_height_pixels=128,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    ).fork(2)
    nwp_dp, nwp_t0_dp = SelectSpatialSlicePixels(
        nwp_dp,
        location_datapipe=location_dp3,
        roi_width_pixels=64,
        roi_height_pixels=64,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
    ).fork(2)
    nwp_dp = Downsample(nwp_dp, y_coarsen=16, x_coarsen=16)
    nwp_t0_dp = SelectLiveT0Time(nwp_t0_dp, dim_name="init_time_utc")
    nwp_dp = ConvertToNWPTargetTime(
        nwp_dp,
        t0_datapipe=nwp_t0_dp,
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(hours=2),
        forecast_duration=timedelta(hours=3),
    )
    gsp_t0_dp = SelectLiveT0Time(gsp_dp)
    gsp_dp = SelectLiveTimeSlice(
        gsp_dp,
        t0_datapipe=gsp_t0_dp,
        history_duration=timedelta(hours=2),
    )
    sat_t0_dp = SelectLiveT0Time(sat_t0_dp)
    sat_dp = SelectLiveTimeSlice(
        sat_dp,
        t0_datapipe=sat_t0_dp,
        history_duration=timedelta(hours=1),
    )
    passiv_t0_dp = SelectLiveT0Time(pv_t0_dp)
    pv_dp = SelectLiveTimeSlice(
        pv_dp,
        t0_datapipe=passiv_t0_dp,
        history_duration=timedelta(hours=1),
    )
    gsp_dp = GSPIterator(gsp_dp)

    sat_dp = Normalize(sat_dp, mean=SAT_MEAN["HRV"] / 4, std=SAT_STD["HRV"] / 4).map(
        lambda x: x.resample(time_utc="5T").interpolate("linear")
    )  # Interplate to 5 minutes incase its 15 minutes
    nwp_dp = Normalize(nwp_dp, mean=NWP_MEAN, std=NWP_STD)
    topo_dp = Normalize(topo_dp, calculate_mean_std_from_example=True)

    ####################################
    #
    # Equivalent to PP's np_batch_processors
    #
    #####################################

    sat_dp = ConvertSatelliteToNumpyBatch(sat_dp, is_hrv=True)
    sat_dp = ExtendTimestepsToFuture(
        sat_dp,
        forecast_duration=timedelta(hours=2),
        sample_period_duration=timedelta(minutes=5),
    )
    sat_dp = MergeNumpyExamplesToBatch(sat_dp, n_examples_per_batch=4)
    pv_dp = ConvertPVToNumpyBatch(pv_dp)
    pv_dp = ExtendTimestepsToFuture(
        pv_dp,
        forecast_duration=timedelta(hours=2),
        sample_period_duration=timedelta(minutes=5),
    )
    pv_dp = MergeNumpyExamplesToBatch(pv_dp, n_examples_per_batch=4)
    gsp_dp = ConvertGSPToNumpyBatch(gsp_dp)
    gsp_dp = ExtendTimestepsToFuture(
        gsp_dp,
        forecast_duration=timedelta(hours=8),
        sample_period_duration=timedelta(minutes=30),
    )
    gsp_dp = MergeNumpyExamplesToBatch(gsp_dp, n_examples_per_batch=4)
    # Don't need to do NWP as it does go into the future
    nwp_dp = ConvertNWPToNumpyBatch(nwp_dp)
    nwp_dp = MergeNumpyExamplesToBatch(nwp_dp, n_examples_per_batch=4)
    combined_dp = MergeNumpyModalities([gsp_dp, pv_dp, sat_dp, nwp_dp])

    combined_dp = AlignGSPto5Min(
        combined_dp, batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc
    )
    combined_dp = EncodeSpaceTime(combined_dp)
    combined_dp = SaveT0Time(combined_dp)
    combined_dp = AddSunPosition(combined_dp, modality_name="hrvsatellite")
    combined_dp = AddSunPosition(combined_dp, modality_name="pv")
    combined_dp = AddSunPosition(combined_dp, modality_name="gsp")
    combined_dp = AddSunPosition(combined_dp, modality_name="gsp_5_min")
    combined_dp = AddSunPosition(combined_dp, modality_name="nwp_target_time")
    combined_dp = AddTopographicData(combined_dp, topo_dp)
    combined_dp = SetSystemIDsToOne(combined_dp)

    batch = next(iter(combined_dp))

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


def test_power_perceiver_production_functional(sat_hrv_dp, passiv_dp, topo_dp, gsp_dp, nwp_dp):
    ####################################
    #
    # Equivalent to PP's loading and filtering methods
    #
    #####################################
    # Normalize GSP and PV on whole dataset here

    gsp_dp = gsp_dp.normalize(
        normalize_fn=lambda x: x / x.capacity_mwp
    ).add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=2)
    )
    location_dp1, location_dp2, location_dp3 = gsp_dp.location_picker(
        return_all_locations=True
    ).fork(3)

    passiv_dp, pv_t0_dp = (
        passiv_dp.normalize(normalize_fn=lambda x: x / x.capacity_wp)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
        )
        .select_spatial_slice_meters(
            location_datapipe=location_dp1, roi_width_meters=960_000, roi_height_meters=960_000
        )
        .ensure_n_pv_systems_per_example(n_pv_systems_per_example=8)
        .fork(2)
    )
    topo_dp = topo_dp.reproject_topography().normalize(calculate_mean_std_from_example=True)
    sat_hrv_dp, sat_t0_dp = (
        sat_hrv_dp.convert_satellite_to_int8()
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
        )
        .select_spatial_slice_pixels(
            location_datapipe=location_dp2,
            roi_width_pixels=256,
            roi_height_pixels=128,
            y_dim_name="y_geostationary",
            x_dim_name="x_geostationary",
        )
        .fork(2)
    )

    nwp_dp, nwp_t0_dp = (
        nwp_dp.add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2)
        )
        .select_spatial_slice_pixels(
            location_datapipe=location_dp3,
            roi_width_pixels=64,
            roi_height_pixels=64,
            y_dim_name="y_osgb",
            x_dim_name="x_osgb",
        )
        .downsample(y_coarsen=16, x_coarsen=16)
        .fork(2)
    )

    nwp_t0_dp = nwp_t0_dp.select_live_t0_time(dim_name="init_time_utc")
    gsp_t0_dp = gsp_dp.select_live_t0_time()
    sat_t0_dp = sat_t0_dp.select_live_t0_time()
    pv_t0_dp = pv_t0_dp.select_live_t0_time()

    gsp_dp = (
        gsp_dp.select_live_time_slice(t0_datapipe=gsp_t0_dp, history_duration=timedelta(hours=2))
        .gsp_iterator()
        .convert_gsp_to_numpy_batch()
        .extend_timesteps_to_future(
            forecast_duration=timedelta(hours=8),
            sample_period_duration=timedelta(minutes=30),
        )
        .merge_numpy_examples_to_batch(n_examples_per_batch=4)
    )
    sat_hrv_dp = (
        sat_hrv_dp.select_live_time_slice(
            t0_datapipe=sat_t0_dp,
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
    passiv_dp = (
        passiv_dp.select_live_time_slice(
            t0_datapipe=pv_t0_dp,
            history_duration=timedelta(hours=1),
        )
        .convert_pv_to_numpy_batch()
        .extend_timesteps_to_future(
            forecast_duration=timedelta(hours=2),
            sample_period_duration=timedelta(minutes=5),
        )
        .merge_numpy_examples_to_batch(n_examples_per_batch=4)
    )
    nwp_dp = (
        nwp_dp.convert_to_nwp_target_time(
            t0_datapipe=nwp_t0_dp,
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
    combined_dp = (
        MergeNumpyModalities([gsp_dp, passiv_dp, sat_hrv_dp, nwp_dp])
        .align_gsp_to_5_min(batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc)
        .encode_space_time()
        .save_t0_time()
        .add_sun_position(modality_name="hrvsatellite")
        .add_sun_position(modality_name="pv")
        .add_sun_position(modality_name="gsp")
        .add_sun_position(modality_name="gsp_5_min")
        .add_sun_position(modality_name="nwp_target_time")
        .add_topographic_data(topo_dp)
        .set_system_ids_to_one()
        .ensure_n_nwp_variables(num_variables=10)
    )

    batch = next(iter(combined_dp))

    assert len(batch[BatchKey.hrvsatellite_time_utc]) == 4
    assert len(batch[BatchKey.hrvsatellite_time_utc][0]) == 37
    assert len(batch[BatchKey.nwp_target_time_utc][0]) == 6
    assert len(batch[BatchKey.nwp_init_time_utc][0]) == 6
    assert len(batch[BatchKey.pv_time_utc][0]) == 37
    assert len(batch[BatchKey.gsp_time_utc][0]) == 21

    assert batch[BatchKey.hrvsatellite_actual].shape == (4, 13, 1, 128, 256)
    assert batch[BatchKey.nwp].shape == (4, 6, 10, 4, 4)
    assert batch[BatchKey.pv].shape == (4, 13, 8)
    assert batch[BatchKey.gsp].shape == (4, 5, 1)
    assert batch[BatchKey.hrvsatellite_surface_height].shape == (4, 128, 256)
