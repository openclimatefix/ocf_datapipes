from datetime import timedelta

from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.transform.numpy import (
    AddSunPosition,
    AlignGSPto5Min,
    EncodeSpaceTime,
    ExtendTimestepsToFuture,
    SaveT0Time,
)
from ocf_datapipes.utils.consts import BatchKey


def test_add_sun_position_all(all_loc_np_datapipe):
    combined_datapipe = AlignGSPto5Min(
        all_loc_np_datapipe, batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc
    )
    combined_datapipe = EncodeSpaceTime(combined_datapipe)
    combined_datapipe = SaveT0Time(combined_datapipe)
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="hrvsatellite")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="pv")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="gsp")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="gsp_5_min")
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="nwp_target_time")
    data = next(iter(combined_datapipe))


def test_add_sun_position_satellite(sat_datapipe):
    pass


def test_add_sun_position_gsp(gsp_datapipe):
    pass


def test_add_sun_position_passiv(passiv_datapipe):
    pass


def test_add_sun_position_nwp(nwp_datapipe):
    pass
