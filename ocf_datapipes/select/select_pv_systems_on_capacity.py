"""Select PV systems based off their capacity"""
from typing import Union

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("select_pv_systems_on_capacity")
class SelectPVSystemsOnCapacityIterDataPipe(IterDataPipe):
    """Select PV systems based off their capacity"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        min_capacity_watts: Union[int, float] = 0.0,
        max_capacity_watts: Union[int, float] = np.inf,
    ):
        """
        Select PV systems based off their capacity

        Args:
            source_datapipe: Datapipe of PV data
            min_capacity_watts: Min capacity in watts
            max_capacity_watts: Max capacity in watts
        """
        self.source_datapipe = source_datapipe
        self.min_capacity_watts = min_capacity_watts
        self.max_capaciity_watts = max_capacity_watts

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            # Drop based off capacity here
            # TODO Do
            yield xr_data


"""

# Drop any PV systems whose PV capacity is too low:
    PV_CAPACITY_THRESHOLD_W = 100
    pv_systems_to_drop =
    pv_capacity_watt_power.index[pv_capacity_watt_power <= PV_CAPACITY_THRESHOLD_W]
    pv_systems_to_drop = pv_systems_to_drop.intersection(pv_power_watts.columns)
    _log.info(
        f"Dropping {len(pv_systems_to_drop)} PV systems because their max power is less than"
        f" {PV_CAPACITY_THRESHOLD_W}"
    )
    pv_power_watts.drop(columns=pv_systems_to_drop, inplace=True)

"""
