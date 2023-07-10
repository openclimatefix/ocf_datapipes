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
    DropGSP,
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
    CreatePVImage,
    Downsample,
    EnsureNPVSystemsPerExample,
    Normalize,
    PreProcessMetNet,
    ReprojectTopography,
)
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD, SAT_MEAN, SAT_STD, BatchKey

import pytest


# N.B First change which broke this test was changing the NWP data in the test directory to include
# more forecast steps
@pytest.mark.skip(reason="Not maintained for the moment")
def test_metnet_production(
    sat_hrv_datapipe, sat_datapipe, passiv_datapipe, topo_datapipe, gsp_datapipe, nwp_datapipe
):
    ####################################
    #
    # Equivalent to PP's loading and filtering methods
    #
    #####################################
    # Normalize GSP and PV on whole dataset here
    pv_datapipe = passiv_datapipe
    gsp_datapipe, gsp_loc_datapipe = DropGSP(gsp_datapipe, gsps_to_keep=[0]).fork(2)
    gsp_datapipe = Normalize(
        gsp_datapipe, normalize_fn=lambda x: x / x.installed_capacity_megawatt_power
    )
    topo_datapipe = ReprojectTopography(topo_datapipe)
    sat_hrv_datapipe = AddT0IdxAndSamplePeriodDuration(
        sat_hrv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )
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

    (
        location_datapipe1,
        location_datapipe2,
        location_datapipe3,
        location_datapipe4,
        location_datapipe5,
    ) = LocationPicker(gsp_loc_datapipe, return_all_locations=True).fork(
        5
    )  # Its in order then
    pv_datapipe, pv_t0_datapipe = SelectSpatialSliceMeters(
        pv_datapipe,
        location_datapipe=location_datapipe1,
        roi_width_meters=100_000,
        roi_height_meters=100_000,
    ).fork(
        2
    )  # Has to be large as test PV systems aren't in first 20 GSPs it seems
    nwp_datapipe, nwp_t0_datapipe = Downsample(nwp_datapipe, y_coarsen=16, x_coarsen=16).fork(2)
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
    sat_t0_datapipe = SelectLiveT0Time(sat_datapipe)
    sat_datapipe, image_datapipe = SelectLiveTimeSlice(
        sat_datapipe,
        t0_datapipe=sat_t0_datapipe,
        history_duration=timedelta(hours=1),
    ).fork(2)
    sat_hrv_t0_datapipe = SelectLiveT0Time(sat_hrv_datapipe)
    sat_hrv_datapipe = SelectLiveTimeSlice(
        sat_hrv_datapipe,
        t0_datapipe=sat_hrv_t0_datapipe,
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

    pv_datapipe = CreatePVImage(pv_datapipe, image_datapipe)

    sat_hrv_datapipe = Normalize(
        sat_hrv_datapipe, mean=SAT_MEAN["HRV"] / 4, std=SAT_STD["HRV"] / 4
    ).map(
        lambda x: x.resample(time_utc="5T").interpolate("linear")
    )  # Interplate to 5 minutes incase its 15 minutes
    sat_datapipe = Normalize(sat_datapipe, mean=SAT_MEAN["IR_016"], std=SAT_STD["IR_016"]).map(
        lambda x: x.resample(time_utc="5T").interpolate("linear")
    )  # Interplate to 5 minutes incase its 15 minutes
    nwp_datapipe = Normalize(nwp_datapipe, mean=NWP_MEAN, std=NWP_STD)
    topo_datapipe = Normalize(topo_datapipe, calculate_mean_std_from_example=True)

    # Now combine in the MetNet format
    combined_datapipe = PreProcessMetNet(
        [
            nwp_datapipe,
            sat_hrv_datapipe,
            sat_datapipe,
            pv_datapipe,
        ],
        location_datapipe=location_datapipe5,
        center_width=500_000,
        center_height=1_000_000,
        context_height=10_000_000,
        context_width=10_000_000,
        output_width_pixels=512,
        output_height_pixels=512,
        add_sun_features=True,
    )

    batch = next(iter(combined_datapipe))
    assert ~np.isnan(batch).any()
    print(batch.shape)
    batch = next(iter(gsp_datapipe))
    print(batch.shape)
