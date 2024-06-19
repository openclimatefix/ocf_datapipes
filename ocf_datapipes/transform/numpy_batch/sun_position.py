"""Datapipes to add Sun position to NumpyBatch"""

import warnings

import numpy as np
import pvlib
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey
from ocf_datapipes.utils.consts import (
    AZIMUTH_MEAN,
    AZIMUTH_STD,
    ELEVATION_MEAN,
    ELEVATION_STD,
)
from ocf_datapipes.utils.geospatial import osgb_to_lon_lat


def _get_azimuth_and_elevation(lon, lat, dt, must_be_finite):
    if type(dt[0]) == np.datetime64:
        # This caused an issue if it was 'datetime64[s]'
        dt = np.array(dt, dtype="datetime64[ns]")

    if not np.isfinite([lon, lat]).all():
        if must_be_finite:
            raise ValueError(f"Non-finite (lon, lat) = ({lon}, {lat}")
        return (
            np.full_like(dt, fill_value=np.nan).astype(np.float32),
            np.full_like(dt, fill_value=np.nan).astype(np.float32),
        )

    else:
        solpos = pvlib.solarposition.get_solarposition(
            time=dt,
            latitude=lat,
            longitude=lon,
            # Which `method` to use?
            # pyephem seemed to be a good mix between speed and ease
            # but causes segfaults!
            # nrel_numba doesn't work when using multiple worker processes.
            # nrel_c is probably fastest but requires C code to be
            #   manually compiled: https://midcdmz.nrel.gov/spa/
        )
        azimuth = solpos["azimuth"]
        elevation = solpos["elevation"]
        return azimuth, elevation


@functional_datapipe("add_sun_position")
class AddSunPositionIterDataPipe(IterDataPipe):
    """Adds the sun position to the NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe, modality_name: str):
        """
        Adds the sun position to the NumpyBatch

        Args:
            source_datapipe: Datapipe of NumpyBatch
            modality_name: Modality to add the sun position for
        """
        self.source_datapipe = source_datapipe
        self.modality_name = modality_name
        assert self.modality_name in [
            "hrvsatellite",
            "gsp",
            "pv",
            "wind",
        ], f"Cant add sun position on {self.modality_name}"

    def __iter__(self):
        for np_batch in self.source_datapipe:
            if self.modality_name == "hrvsatellite":
                # TODO Make work with Lat/Lons instead
                y_osgb = np_batch[BatchKey.hrvsatellite_y_osgb]  # Shape: optional[example], y, x
                x_osgb = np_batch[BatchKey.hrvsatellite_x_osgb]  # Shape: optional[example], y, x
                time_utc = np_batch[
                    BatchKey.hrvsatellite_time_utc
                ]  # Shape: optional[example], time

                # Get the time and position for the centre of the t0 frame:
                y_centre_idx = int(y_osgb.shape[-2] // 2)
                x_centre_idx = int(y_osgb.shape[-1] // 2)
                y_osgb = y_osgb[..., y_centre_idx, x_centre_idx]  # Shape: optional[example]
                x_osgb = x_osgb[..., y_centre_idx, x_centre_idx]  # Shape: optional[example]

            elif self.modality_name == "pv":
                lats = np_batch[BatchKey.pv_latitude]  # Shape: (optional[example], n_pvs)
                lons = np_batch[BatchKey.pv_longitude]  # Shape: (optional[example], n_pvs)
                time_utc = np_batch[BatchKey.pv_time_utc]  # Shape: optional[example]

                # If using multiple PV systems, take mean location
                if lats.shape[-1] > 1:
                    # Sometimes, the PV coords can all be NaNs if there are no PV systems
                    # for that datetime and location. e.g. in northern Scotland!
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action="ignore", category=RuntimeWarning, message="Mean of empty slice"
                        )
                        lats = np.nanmean(lats, axis=-1)
                        lons = np.nanmean(lons, axis=-1)
                else:
                    lats = lats[..., 0]
                    lons = lons[..., 0]

            elif self.modality_name == "wind":
                lats = np_batch[BatchKey.wind_latitude]  # Shape: (optional[example], n_pvs)
                lons = np_batch[BatchKey.wind_longitude]  # Shape: (optional[example], n_pvs)
                time_utc = np_batch[BatchKey.wind_time_utc]  # Shape: optional[example]

                # If using multiple PV systems, take mean location
                if lats.shape[-1] > 1:
                    # Sometimes, the PV coords can all be NaNs if there are no PV systems
                    # for that datetime and location. e.g. in northern Scotland!
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action="ignore", category=RuntimeWarning, message="Mean of empty slice"
                        )
                        lats = np.nanmean(lats, axis=-1)
                        lons = np.nanmean(lons, axis=-1)
                else:
                    lats = lats[..., 0]
                    lons = lons[..., 0]

            elif self.modality_name == "gsp":
                y_osgb = np_batch[BatchKey.gsp_y_osgb]  # Shape: optional[example], n_gsps
                x_osgb = np_batch[BatchKey.gsp_x_osgb]  # Shape: optional[example], n_gsps
                time_utc = np_batch[BatchKey.gsp_time_utc]  # Shape: optional[example],

                # If using multiple GSPs, take mean location
                if x_osgb.shape[-1] > 1:
                    y_osgb = np.nanmean(y_osgb, axis=-1)
                    x_osgb = np.nanmean(x_osgb, axis=-1)
                else:
                    y_osgb = y_osgb[..., 0]
                    x_osgb = x_osgb[..., 0]

            else:
                raise ValueError(f"Unrecognized modality: {self.modality_name }")

            # As we move away from OSGB and towards lon, lat we can exclude more sources here
            if self.modality_name not in ["pv", "wind"]:
                # Convert to the units that pvlib expects: lon, lat
                lons, lats = osgb_to_lon_lat(x=x_osgb, y=y_osgb)

            # Elevations must be finite and non-nan except for PV data where values may be missing
            must_be_finite = self.modality_name != "pv"

            # Check if the input is batched
            # time_utc could have shape (batch_size, n_times) or (n_times,)
            assert len(time_utc.shape) in [1, 2]
            is_batched = len(time_utc.shape) == 2

            times = time_utc.astype("datetime64[s]")

            if is_batched:
                assert lons.shape == (time_utc.shape[0],)
                assert lats.shape == (time_utc.shape[0],)

                azimuth = np.full_like(time_utc, fill_value=np.nan).astype(np.float32)
                elevation = np.full_like(time_utc, fill_value=np.nan).astype(np.float32)

                # Loop round each example to get the Sun's elevation and azimuth
                for example_idx, (lon, lat, dt) in enumerate(zip(lons, lats, times)):
                    azimuth[example_idx], elevation[example_idx] = _get_azimuth_and_elevation(
                        lon, lat, dt, must_be_finite
                    )
            else:
                assert (isinstance(lons, np.ndarray) and lons.shape == ()) or isinstance(
                    lons, float
                )
                assert (isinstance(lats, np.ndarray) and lats.shape == ()) or isinstance(
                    lats, float
                )

                azimuth, elevation = _get_azimuth_and_elevation(lons, lats, times, must_be_finite)

            # Normalize
            azimuth = (azimuth - AZIMUTH_MEAN) / AZIMUTH_STD
            elevation = (elevation - ELEVATION_MEAN) / ELEVATION_STD

            # Store
            azimuth_batch_key = BatchKey[self.modality_name + "_solar_azimuth"]
            elevation_batch_key = BatchKey[self.modality_name + "_solar_elevation"]
            np_batch[azimuth_batch_key] = azimuth
            np_batch[elevation_batch_key] = elevation

            yield np_batch
