"""Stacking Xarray objects to Numpy inputs"""
from typing import List

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils import Zipper


@functional_datapipe("stack_xarray")
class StackXarrayIterDataPipe(IterDataPipe):
    """Stack Xarray datasets together"""

    def __init__(
        self,
        source_datapipes: List[IterDataPipe],
    ):
        """

        Processes set of Xarray datasets into Numpy

        Args:
            source_datapipes: Datapipes that emit xarray datasets
                with latitude/longitude coordinates included
        """
        self.source_datapipes = source_datapipes

    def __iter__(self) -> np.ndarray:
        for xr_datas in Zipper(*self.source_datapipes):
            stack = []
            for xr_index, xr_data in enumerate(xr_datas):
                #print(xr_data.dims)
                #print(xr_data.shape)
                if "channel" in xr_data.dims:
                    channel_idx = xr_data.dims.index("channel")
                else:
                    channel_idx = -1
                if channel_idx != -1 and channel_idx > 0:
                    xr_data = xr_data.transpose("channel", *xr_data.dims[:channel_idx], *xr_data.dims[channel_idx+1:])
                    channel_idx = 0
                #print(xr_data.dims)
                # Resamples to the same number of pixels for both center and contexts
                xr_data = xr_data.to_numpy()
                #print(xr_data.shape)
                if len(xr_data.shape) == 2:  # Need to add channel dimension
                    xr_data = np.expand_dims(xr_data, axis=0)
                if len(xr_data.shape) == 3:  # Need to add channel dimension
                    xr_data = np.expand_dims(xr_data, axis=0)
                #print(xr_data.shape)
                stack.append(xr_data)
            # Pad out time dimension to be the same, using the largest one
            # All should have 4 dimensions at this point
            max_time_len = np.max([c.shape[1] for c in stack])
            for i in range(len(stack)):
                stack[i] = np.pad(
                    stack[i],
                    pad_width=(
                        (0, 0),
                        (0, max_time_len - stack[i].shape[1]),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0.0,
                )

            stacked_data = np.concatenate([*stack], axis=0)
            yield stacked_data
