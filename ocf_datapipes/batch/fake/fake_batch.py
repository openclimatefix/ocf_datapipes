""" Make fake batch """
from datetime import datetime, timezone

from ocf_datapipes.batch.fake.gsp import make_fake_gsp_data
from ocf_datapipes.batch.fake.nwp import make_fake_nwp_data
from ocf_datapipes.batch.fake.pv import make_fake_pv_data
from ocf_datapipes.batch.fake.satellite import make_fake_satellite_data
from ocf_datapipes.batch.fake.sun import make_fake_sun_data
from ocf_datapipes.config.model import Configuration


def make_fake_batch(configuration: Configuration) -> dict:
    """
    Make a random fake batch, this is useful for models that use this object

    Args:
        configuration: a configuration file

    Returns: dictionary containing the batch

    """

    # time now rounded down to 5 minutes
    t0_datetime_utc = datetime.now(tz=timezone.utc)
    t0_datetime_utc = t0_datetime_utc.replace(minute=t0_datetime_utc.minute // 5 * 5)
    t0_datetime_utc = t0_datetime_utc.replace(second=0)
    t0_datetime_utc = t0_datetime_utc.replace(microsecond=0)

    # make fake PV data
    batch_pv = make_fake_pv_data(configuration=configuration, t0_datetime_utc=t0_datetime_utc)

    # make NWP data
    batch_nwp = make_fake_nwp_data(configuration=configuration, t0_datetime_utc=t0_datetime_utc)

    # make GSP data
    batch_gsp = make_fake_gsp_data(configuration=configuration, t0_datetime_utc=t0_datetime_utc)

    # make hrv and normal satellite data
    batch_satellite = make_fake_satellite_data(
        configuration=configuration, t0_datetime_utc=t0_datetime_utc, is_hrv=False
    )
    batch_hrv_satellite = make_fake_satellite_data(
        configuration=configuration, t0_datetime_utc=t0_datetime_utc, is_hrv=True
    )

    # make sun features
    batch_sun = make_fake_sun_data(configuration=configuration)

    batch = {
        **batch_pv,
        **batch_nwp,
        **batch_gsp,
        **batch_satellite,
        **batch_hrv_satellite,
        **batch_sun,
    }

    return batch
