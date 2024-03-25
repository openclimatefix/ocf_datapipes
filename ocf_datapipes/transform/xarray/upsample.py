"""Up Sample Xarray datasets Datapipe"""

import logging
from typing import Optional

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

log = logging.getLogger(__name__)


@functional_datapipe("upsample")
class UpSampleIterDataPipe(IterDataPipe):
    """Up Sample Xarray dataset with Interpolate"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        y_upsample: int,
        x_upsample: int,
        x_dim_name: str = "longitude",
        y_dim_name: str = "latitude",
        keep_same_shape: bool = False,
        round_to_dp: Optional[int] = None,
    ):
        """
        Up Sample xarray dataset/dataarrays with interpolate

        Args:
            source_datapipe: Datapipe emitting Xarray dataset
            y_upsample: up sample value in the y direction
            x_upsample: Up sample value in the x direction
            x_dim_name: X dimension name
            y_dim_name: Y dimension name
            keep_same_shape: Optional to keep the same shape. Defaults to zero.
                If True, shape is trimmed around the edges.
            round_to_dp: Try to round values to this number of decimal places.
                Default is None, so no rounding is done.
        """
        self.source_datapipe = source_datapipe
        self.y_upsample = y_upsample
        self.x_upsample = x_upsample
        self.x_dim_name = x_dim_name
        self.y_dim_name = y_dim_name
        self.keep_same_shape = keep_same_shape
        self.round_to_dp = round_to_dp

    def __iter__(self):
        """Coarsen the data on the specified dimensions"""
        for xr_data in self.source_datapipe:
            log.info("Up Sampling Data")

            # get current x and y values
            current_x_dim_values = getattr(xr_data, self.x_dim_name).values
            current_y_dim_values = getattr(xr_data, self.y_dim_name).values

            # get current interval values
            current_x_interval = np.abs(current_x_dim_values[1] - current_x_dim_values[0])
            current_y_interval = np.abs(current_y_dim_values[1] - current_y_dim_values[0])

            # new intervals
            new_x_interval = current_x_interval / self.x_upsample
            new_y_interval = current_y_interval / self.y_upsample

            if self.round_to_dp is not None:
                new_x_interval = np.round(new_x_interval, self.round_to_dp)
                new_y_interval = np.round(new_y_interval, self.round_to_dp)

            if self.keep_same_shape:
                # up sample the center of the image and keep the same shape as original image

                center_x = current_x_dim_values[int(len(current_x_dim_values) / 2)]
                center_y = current_y_dim_values[int(len(current_y_dim_values) / 2)]

                new_x_min = center_x - new_x_interval * int(len(current_x_dim_values) / 2)
                new_x_max = new_x_min + new_x_interval * (len(current_x_dim_values) - 1)

                new_y_min = center_y - new_y_interval * int(len(current_y_dim_values) / 2)
                new_y_max = new_y_min + new_y_interval * (len(current_y_dim_values) - 1)

            else:
                new_x_min = min(current_x_dim_values)
                new_x_max = max(current_x_dim_values)

                new_y_min = min(current_y_dim_values)
                new_y_max = max(current_y_dim_values)

            # round to decimals places
            if self.round_to_dp is not None:
                new_x_min = np.round(new_x_min, self.round_to_dp)
                new_x_max = np.round(new_x_max, self.round_to_dp)

                new_y_min = np.round(new_y_min, self.round_to_dp)
                new_y_max = np.round(new_y_max, self.round_to_dp)

            # get new x values
            new_x_dim_values = list(
                np.arange(
                    new_x_min,
                    new_x_max + new_x_interval,
                    new_x_interval,
                )
            )

            # get new y values
            new_y_dim_values = list(
                np.arange(
                    new_y_min,
                    new_y_max + new_y_interval,
                    new_y_interval,
                )
            )

            # check the order
            if current_x_dim_values[0] > current_x_dim_values[1]:
                new_x_dim_values = new_x_dim_values[::-1]
                new_x_max, new_x_min = new_x_min, new_x_max
                log.debug("X dims are reversed")
            if current_y_dim_values[0] > current_y_dim_values[1]:
                new_y_dim_values = new_y_dim_values[::-1]
                new_y_max, new_y_min = new_y_min, new_y_max
                log.debug("Y dims are reversed")

            log.info(
                f"Up Sampling X from ({min(current_x_dim_values)}, {current_x_interval}, "
                f"{max(current_x_dim_values)}) to ({new_x_min}, {new_x_interval}, {new_x_max})"
            )
            log.info(
                f"Up Sampling Y from ({min(current_y_dim_values)}, {current_y_interval}, "
                f"{max(current_y_dim_values)}) to ({new_y_min}, {new_y_interval}, {new_y_max})"
            )

            # resample
            xr_data = xr_data.interp(**{self.x_dim_name: new_x_dim_values})
            xr_data = xr_data.interp(**{self.y_dim_name: new_y_dim_values})

            yield xr_data
