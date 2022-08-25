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


def test_add_sun_position_all(all_loc_np_dp):
    combined_dp = AlignGSPto5Min(
        all_loc_np_dp, batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc
    )
    combined_dp = EncodeSpaceTime(combined_dp)
    combined_dp = SaveT0Time(combined_dp)
    combined_dp = AddSunPosition(combined_dp, modality_name="hrvsatellite")
    combined_dp = AddSunPosition(combined_dp, modality_name="pv")
    combined_dp = AddSunPosition(combined_dp, modality_name="gsp")
    combined_dp = AddSunPosition(combined_dp, modality_name="gsp_5_min")
    # TODO Add NWP target time here
    # TODO Need a locationPicker here to bring down to a few specific locations, GSP has too many locations for this
    data = next(iter(combined_dp))


def test_add_sun_position_satellite(sat_dp):
    pass


def test_add_sun_position_gsp(gsp_dp):
    pass


def test_add_sun_position_passiv(passiv_dp):
    pass


def test_add_sun_position_nwp(nwp_dp):
    pass
