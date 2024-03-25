""" Make fake batch """

from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data.datapipes.iter import IterableWrapper

from ocf_datapipes.batch.fake.gsp import make_fake_gsp_data
from ocf_datapipes.batch.fake.nwp import make_fake_nwp_data
from ocf_datapipes.batch.fake.pv import make_fake_pv_data
from ocf_datapipes.batch.fake.satellite import make_fake_satellite_data
from ocf_datapipes.batch.fake.sun import make_fake_sun_data
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils.utils import datetime64_to_float


def make_fake_batch(
    configuration: Configuration,
    batch_size: int = 8,
    to_torch: Optional[bool] = False,
) -> dict:
    """
    Make a random fake batch, this is useful for models that use this object

    Args:
        configuration: a configuration file
        batch_size: the batch size
        to_torch: optional if we return the batch with torch.Tensor

    Returns: dictionary containing the batch

    """

    # time now rounded down to 5 minutes
    t0_datetime_utc = datetime.now(tz=timezone.utc)
    t0_datetime_utc = t0_datetime_utc.replace(minute=t0_datetime_utc.minute // 5 * 5)
    t0_datetime_utc = t0_datetime_utc.replace(second=0)
    t0_datetime_utc = t0_datetime_utc.replace(microsecond=0)

    # make fake PV data
    batch_pv = make_fake_pv_data(configuration, t0_datetime_utc, batch_size)

    # make NWP data
    batch_nwp = make_fake_nwp_data(configuration, t0_datetime_utc, batch_size)

    # make GSP data
    batch_gsp = make_fake_gsp_data(configuration, t0_datetime_utc, batch_size)

    # make hrv and normal satellite data
    batch_satellite = make_fake_satellite_data(
        configuration,
        t0_datetime_utc,
        is_hrv=False,
        batch_size=batch_size,
    )
    batch_hrv_satellite = make_fake_satellite_data(
        configuration,
        t0_datetime_utc,
        is_hrv=True,
        batch_size=batch_size,
    )

    # make sun features
    batch_sun = make_fake_sun_data(configuration, batch_size)

    batch = {
        **batch_pv,
        **batch_nwp,
        **batch_gsp,
        **batch_satellite,
        **batch_hrv_satellite,
        **batch_sun,
    }

    if to_torch:
        for k, v in batch.items():
            if isinstance(v, int):
                batch[k] = torch.IntTensor(v)
            elif isinstance(v, np.ndarray):
                if v.dtype == "datetime64[s]":
                    batch[k] = torch.from_numpy(datetime64_to_float(v))
                else:
                    batch[k] = torch.from_numpy(v)

    return batch


def fake_data_pipeline(configuration: Union[str, Configuration], batch_size: int = 8):
    """
    Make a fake data pipeline

    Args:
        configuration: a configuration file
        batch_size: Integer batch size to create

    """

    if isinstance(configuration, str):
        configuration = load_yaml_configuration(configuration)

    batch = make_fake_batch(configuration=configuration, to_torch=True, batch_size=batch_size)

    def fake_iter():
        while True:
            yield batch

    return IterableWrapper(fake_iter())
