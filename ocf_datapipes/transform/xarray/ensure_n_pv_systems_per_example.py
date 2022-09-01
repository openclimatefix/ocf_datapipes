"""Ensure there is N PV systems per example"""
import logging

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("ensure_n_pv_systems_per_example")
class EnsureNPVSystemsPerExampleIterDataPipe(IterDataPipe):
    """Ensure there is N PV systems per example"""

    def __init__(self, source_datapipe: IterDataPipe, n_pv_systems_per_example: int, seed=None):
        """
        Ensure there is N PV systems per example

        Args:
            source_datapipe: Datapipe of PV data
            n_pv_systems_per_example: Number of PV systems to have in example
            seed: Random seed for choosing
        """
        self.source_datapipe = source_datapipe
        self.n_pv_systems_per_example = n_pv_systems_per_example
        self.rng = np.random.default_rng(seed=seed)

    def __iter__(self):
        for xr_data in self.source_datapipe:
            if len(xr_data.pv_system_id) > self.n_pv_systems_per_example:
                logger.debug(f"Reducing PV systems to  {self.n_pv_systems_per_example}")
                # More PV systems are available than we need. Reduce by randomly sampling:
                subset_of_pv_system_ids = self.rng.choice(
                    xr_data.pv_system_id,
                    size=self.n_pv_systems_per_example,
                    replace=False,
                )
                xr_data = xr_data.sel(pv_system_id=subset_of_pv_system_ids)
            elif len(xr_data.pv_system_id) < self.n_pv_systems_per_example:
                logger.debug("Padding out PV systems")
                # If we just used `choice(replace=True)` then there's a high chance
                # that the output won't include every available PV system but instead
                # will repeat some PV systems at the expense of leaving some on the table.
                # TODO: Don't repeat PV systems. Pad with NaNs and mask the loss. Issue #73.
                assert len(xr_data.pv_system_id) > 0, (
                    "There are no PV systems at all. " "We need at least one in an example"
                )
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
