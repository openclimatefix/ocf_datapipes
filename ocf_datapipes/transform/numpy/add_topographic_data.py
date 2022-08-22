import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.consts import BatchKey
from ocf_datapipes.utils import NumpyBatch


@functional_datapipe("add_topographic_data")
class AddTopographicDataIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, topo_dp: IterDataPipe):
        self.source_dp = source_dp
        self.topo_dp = topo_dp

    def __iter__(self) -> NumpyBatch:
        for topo in self.topo_dp:
            for np_batch in self.source_dp:
                if BatchKey.hrvsatellite_x_geostationary in np_batch:
                    # Recreate an xr.DataArray of the satellite data. This is required so we can
                    # use xr.combine_by_coords to align the topo data with the satellite data.
                    hrvsatellite_data_array = xr.DataArray(
                        # We're not actually interested in the image. But xarray won't make an
                        # empty DataArray without data in the right shape.
                        # There's nothing special about the x_osgb data. It's just convenient because
                        # it's the right shape!
                        np_batch[BatchKey.hrvsatellite_x_osgb],
                        dims=("example", "y", "x"),
                        coords={
                            "y_geostationary": (
                                ("example", "y"),
                                np_batch[BatchKey.hrvsatellite_y_geostationary],
                            ),
                            "x_geostationary": (
                                ("example", "x"),
                                np_batch[BatchKey.hrvsatellite_x_geostationary],
                            ),
                        },
                    )
                    hrvsatellite_surface_height = _get_surface_height_for_satellite(
                        surface_height=topo, satellite=hrvsatellite_data_array
                    )
                    np_batch[BatchKey.hrvsatellite_surface_height] = hrvsatellite_surface_height
                yield np_batch


def _get_surface_height_for_satellite(
    surface_height: xr.DataArray, satellite: xr.DataArray
) -> np.ndarray:
    num_examples = satellite.shape[0]
    surface_height = surface_height.rename("surface_height")
    surface_height_for_batch = np.full_like(satellite.values, fill_value=np.NaN)
    for example_idx in range(num_examples):
        satellite_example = satellite.isel(example=example_idx)
        msg = "Satellite imagery must start in the top-left!"
        assert satellite_example.y_geostationary[0] > satellite_example.y_geostationary[-1], msg
        assert satellite_example.x_geostationary[0] < satellite_example.x_geostationary[-1], msg

        satellite_example = satellite_example.rename(
            {"y_geostationary": "y", "x_geostationary": "x"}
        ).rename("sat")
        surface_height_for_example = surface_height.sel(
            y=slice(
                satellite_example.y[0],
                satellite_example.y[-1],
            ),
            x=slice(
                satellite_example.x[0],
                satellite_example.x[-1],
            ),
        )
        # Align by coordinates. This will result in lots of NaNs in the surface height data:
        aligned = xr.combine_by_coords(
            (surface_height_for_example, satellite_example), join="outer"
        )
        # Fill in the NaNs:
        surface_height_for_example = (
            aligned["surface_height"].ffill("x").ffill("y").bfill("x").bfill("y")
        )

        # Now select exactly the same coordinates from the surface height data as the satellite data
        aligned = xr.combine_by_coords((surface_height_for_example, satellite_example), join="left")

        surface_height_for_batch[example_idx] = aligned["surface_height"].values

    # If we slightly ran off the edge of the topo data then we'll get NaNs.
    # TODO: Enlarge topo data so we never get NaNs!
    surface_height_for_batch = np.nan_to_num(surface_height_for_batch, nan=0)

    return surface_height_for_batch
