import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("ensure_n_pv_systems_per_example")
class EnsureNPVSystemsPerExampleIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, n_pv_systems_per_example: int, seed=None):
        self.source_datapipe = source_datapipe
        self.n_pv_systems_per_example = n_pv_systems_per_example
        self.rng = np.random.default_rng(seed=seed)

    def __iter__(self):
        for xr_data in self.source_datapipe:
            if len(xr_data.pv_system_id) > self.n_pv_systems_per_example:
                # More PV systems are available than we need. Reduce by randomly sampling:
                subset_of_pv_system_ids = self.rng.choice(
                    xr_data.pv_system_id,
                    size=self.n_pv_systems_per_example,
                    replace=False,
                )
                xr_data = xr_data.sel(pv_system_id=subset_of_pv_system_ids)
            elif len(xr_data.pv_system_id) < self.n_pv_systems_per_example:
                # If we just used `choice(replace=True)` then there's a high chance
                # that the output won't include every available PV system but instead
                # will repeat some PV systems at the expense of leaving some on the table.
                # TODO: Don't repeat PV systems. Instead, pad with NaNs and mask the loss. Issue #73.
                n_random_pv_systems = self.n_pv_systems_per_example - len(xr_data.pv_system_id)
                allow_replacement = n_random_pv_systems > len(xr_data.pv_system_id)
                random_pv_system_ids = self.rng.choice(
                    xr_data.pv_system_id,
                    size=n_random_pv_systems,
                    replace=allow_replacement,
                )
                xr_data = xr.concat(
                    (xr_data, xr_data.sel(pv_system_id=random_pv_system_ids)),
                    dim="pv_system_id",
                )
            yield xr_data
