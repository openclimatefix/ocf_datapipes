"""Ensure there is N PV systems per example"""

import logging

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("ensure_n_gsp_per_example")
class EnsureNGSPSPerExampleIterDataPipe(IterDataPipe):
    """Ensure there is N GSP per example"""

    def __init__(self, source_datapipe: IterDataPipe, n_gsps_per_example: int, seed=None):
        """
        Ensure there is N GSPS per example

        Args:
            source_datapipe: Datapipe of GSP data
            n_gsps_per_example: Number of GSPS to have in example
            seed: Random seed for choosing
        """
        self.source_datapipe = source_datapipe
        self.n_gsps_per_example = n_gsps_per_example
        self.rng = np.random.default_rng(seed=seed)

    def __iter__(self):
        for xr_data in self.source_datapipe:
            if len(xr_data.gsp_id) > self.n_gsps_per_example:
                logger.debug(f"Reducing GSPS to  {self.n_gsps_per_example}")
                # More PV systems are available than we need. Reduce by randomly sampling:
                subset_of_gsp_ids = self.rng.choice(
                    xr_data.gsp_id,
                    size=self.n_gsps_per_example,
                    replace=False,
                )
                xr_data = xr_data.sel(gsp_id=subset_of_gsp_ids)
            elif len(xr_data.gsp_id) < self.n_gsps_per_example:
                logger.debug("Padding out GSP")
                # If we just used `choice(replace=True)` then there's a high chance
                # that the output won't include every available PV system but instead
                # will repeat some PV systems at the expense of leaving some on the table.
                # TODO: Don't repeat GSP. Pad with NaNs and mask the loss. Issue #73.
                assert len(xr_data.gsp_id) > 0, (
                    "There are no GSPS at all. " "We need at least one in an example"
                )
                n_random_gsps = self.n_gsps_per_example - len(xr_data.gsp_id)
                allow_replacement = n_random_gsps > len(xr_data.gsp_id)
                random_gsp_ids = self.rng.choice(
                    xr_data.gsp_id,
                    size=n_random_gsps,
                    replace=allow_replacement,
                )
                xr_data = xr.concat(
                    (xr_data, xr_data.sel(gsp_id=random_gsp_ids)),
                    dim="gsp_id",
                )
            yield xr_data
