"""Datapipes to add Sun position to NumpyBatch"""
import warnings

import numpy as np
import pandas as pd
import pvlib
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.utils.geospatial import osgb_to_lat_lon

ELEVATION_MEAN = 37.4
ELEVATION_STD = 12.7
AZIMUTH_MEAN = 177.7
AZIMUTH_STD = 41.7


@functional_datapipe("add_sun_position")
class AddSunPositionIterDataPipe(IterDataPipe):
    """Adds the sun position to the NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe, modality_name: str):
        """
        Adds the sun position to the NumpyBatch

        Args:
            source_datapipe: Datapipe of NumpyBatch
            modality_name: modality to add the sun position for
        """
        self.source_datapipe = source_datapipe
        self.modality_name = modality_name
        assert self.modality_name in [
            "hrvsatellite",
            "gsp",
            "gsp_5_min",
            "pv",
            "nwp_target_time",
            "satellite",
        ], f"Cant add sun position on {self.modality_name}"

    def __iter__(self):
        for np_batch in self.source_datapipe:
            # TODO Make work with Lat/Lons instead
            if self.modality_name == "hrvsatellite":
                y_osgb = np_batch[BatchKey.hrvsatellite_y_osgb]  # example, y, x
                x_osgb = np_batch[BatchKey.hrvsatellite_x_osgb]  # example, y, x
                time_utc = np_batch[BatchKey.hrvsatellite_time_utc]  # example, time

                # Get the time and position for the centre of the t0 frame:
                y_centre_idx = int(y_osgb.shape[1] // 2)
                x_centre_idx = int(y_osgb.shape[2] // 2)
                y_osgb_centre = y_osgb[:, y_centre_idx, x_centre_idx]  # Shape: (example,)
                x_osgb_centre = x_osgb[:, y_centre_idx, x_centre_idx]  # Shape: (example,)
            elif self.modality_name == "pv":
                pv_y_osgb = np_batch[BatchKey.pv_y_osgb]
                pv_x_osgb = np_batch[BatchKey.pv_x_osgb]
                time_utc = np_batch[BatchKey.pv_time_utc]
                # Sometimes, the PV coords can all be NaNs if there are no PV systems
                # for that datetime and location. e.g. in northern Scotland!
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action="ignore", category=RuntimeWarning, message="Mean of empty slice"
                    )
                    y_osgb_centre = np.nanmean(pv_y_osgb, axis=1)
                    x_osgb_centre = np.nanmean(pv_x_osgb, axis=1)
            elif self.modality_name == "nwp_target_time":
                y_osgb_centre = np_batch[BatchKey.nwp_y_osgb].mean(axis=-1)
                x_osgb_centre = np_batch[BatchKey.nwp_x_osgb].mean(axis=-1)
                time_utc = np_batch[BatchKey.nwp_target_time_utc]
            else:  # gsp and gsp_5_min:
                y_osgb_centre = np_batch[BatchKey.gsp_y_osgb]
                x_osgb_centre = np_batch[BatchKey.gsp_x_osgb]
                time_utc = np_batch[BatchKey[self.modality_name + "_time_utc"]]

                # Convert to the units that pvlib expects: lat, lon.
            lats, lons = osgb_to_lat_lon(x=x_osgb_centre, y=y_osgb_centre)

            # Loop round each example to get the Sun's elevation and azimuth:
            azimuth = np.full_like(time_utc, fill_value=np.NaN).astype(np.float32)
            elevation = np.full_like(time_utc, fill_value=np.NaN).astype(np.float32)
            must_be_finite = True
            for example_idx, (lat, lon) in enumerate(zip(lats, lons)):
                if not np.isfinite([lat, lon]).all():
                    assert (
                        self.modality_name == "pv"
                    ), f"{self.modality_name} lat and lon must be finite! But {lat=} {lon=}!"
                    # This is PV data, for a location which has no PV systems.
                    must_be_finite = False
                    continue  # Leave azimuth and elevation as NaN for this example_idx.
                dt = pd.to_datetime(time_utc[example_idx], unit="s")
                dt = pd.DatetimeIndex(dt)  # pvlib expects a `pd.DatetimeIndex`.
                solpos = pvlib.solarposition.get_solarposition(
                    time=dt,
                    latitude=lat,
                    longitude=lon,
                    # Which `method` to use?
                    # pyephem seemed to be a good mix between speed and ease but causes segfaults!
                    # nrel_numba doesn't work when using multiple worker processes.
                    # nrel_c is probably fastest but requires C code to be manually compiled:
                    # https://midcdmz.nrel.gov/spa/
                )
                azimuth[example_idx] = solpos["azimuth"]
                elevation[example_idx] = solpos["elevation"]

            # Normalize.
            azimuth = (azimuth - AZIMUTH_MEAN) / AZIMUTH_STD
            elevation = (elevation - ELEVATION_MEAN) / ELEVATION_STD

            # Check
            if must_be_finite:
                assert np.isfinite(azimuth).all()
                assert np.isfinite(elevation).all()

            # Store.
            azimuth_batch_key = BatchKey[self.modality_name + "_solar_azimuth"]
            elevation_batch_key = BatchKey[self.modality_name + "_solar_elevation"]
            np_batch[azimuth_batch_key] = azimuth
            np_batch[elevation_batch_key] = elevation

            yield np_batch
