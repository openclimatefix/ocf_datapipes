"""Create the training/validation datapipe for training the PVNet Model"""
import logging
from typing import Optional, Type, Tuple, List, Union
from datetime import datetime

import xarray as xr
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.iter import IterableWrapper
from ocf_datapipes.convert import (
    ConvertNWPToNumpyBatch,
    ConvertPVToNumpyBatch,
    ConvertSatelliteToNumpyBatch,
    ConvertGSPToNumpyBatch,

)

from ocf_datapipes.batch import MergeNumpyModalities, MergeNWPNumpyModalities
from ocf_datapipes.batch.merge_numpy_examples_to_batch import stack_np_examples_into_batch
from ocf_datapipes.training.common import (
    _get_datapipes_dict,
    check_nans_in_satellite_data,
    concat_xr_time_utc,
    fill_nans_in_arrays,
    fill_nans_in_pv,
    normalize_gsp,
    normalize_pv,
    slice_datapipes_by_time,
    minutes,
    open_and_return_datapipes,
)
from ocf_datapipes.utils.consts import (
    NWP_MEANS,
    NWP_STDS,
    RSS_MEAN,
    RSS_STD,
)
import numpy as np

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils import Location

from ocf_datapipes.utils.geospatial import (
    move_lon_lat_by_meters,
    spatial_coord_type,
)
from ocf_datapipes.select.select_spatial_slice import (
    _get_idx_of_pixel_closest_to_poi, 
    _get_idx_of_pixel_closest_to_poi_geostationary,
    _get_points_from_unstructured_grids,
    convert_coords_to_match_xarray,
    select_spatial_slice_pixels,
)


xr.set_options(keep_attrs=True)
logger = logging.getLogger("pvnet_all_gsp_datapipe")


class SampleFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, sample_list):
        return [self.function(sample) for sample in sample_list]
    
class ZipFunction:
    def __init__(self, function):
        self.function = function
        
    def __call__(self, zipped_sample_list):
        sample_lists = [sample_list for sample_list in zipped_sample_list]
        return [self.function(sample) for sample in zip(*sample_lists)]
            

class SampleRepeat:
    def __init__(self, num_repeats):
        self.num_repeats = num_repeats
        
    def __call__(self, x):
        return [x for _ in range(self.num_repeats)]


class GSPLocationLookup:
    """Query object for GSP location from GSP ID"""

    def __init__(self, x_osgb: xr.DataArray, y_osgb: xr.DataArray):
        """Query object for GSP location from GSP ID

        Args:
            x_osgb: DataArray of the OSGB x-coordinate for any given GSP ID
            y_osgb: DataArray of the OSGB y-coordinate for any given GSP ID

        """
        self.x_osgb = x_osgb
        self.y_osgb = y_osgb

    def __call__(self, gsp_id: int) -> Location:
        """Returns the locations for the input GSP IDs.

        Args:
            gsp_id: Integer ID of the GSP
        """
        return Location(
            x=self.x_osgb.sel(gsp_id=gsp_id).item(),
            y=self.y_osgb.sel(gsp_id=gsp_id).item(),
            id=gsp_id,
        )


    
def create_t0_datapipe(
    datapipes_dict: dict,
    configuration: Configuration,
    shuffle: bool = True,
):
    """
    Takes source datapipes and returns datapipes of appropriate t0 times.

    The t0 times are sampled without replacement.

    Args:
        datapipes_dict: Dictionary of datapipes of input sources for which we want to select
            appropriate location and times.
        configuration: Configuration object for inputs.
        shuffle: Whether to use the internal shuffle function when yielding location times. Else
            location times will be heavily ordered.

    Returns:
        location datapipe, t0 datapipe

    """
    assert "gsp" in datapipes_dict

    contiguous_time_datapipes = []  # Used to store contiguous time periods from each data source

    datapipes_dict["gsp"], key_datapipe = datapipes_dict["gsp"].fork(2, buffer_size=5)

    for key in datapipes_dict.keys():
        if key in ["topo"]:
            continue

        elif key == "nwp":
            for nwp_key in datapipes_dict["nwp"].keys():
                # NWPs are nested since there can be multiple NWP sources
                datapipes_dict["nwp"][nwp_key], datapipe_copy = datapipes_dict["nwp"][nwp_key].fork(
                    2, buffer_size=5
                )

                # Different config setting per NWP source
                nwp_conf = configuration.input_data.nwp[nwp_key]

                if nwp_conf.dropout_timedeltas_minutes is None:
                    max_dropout = minutes(0)
                else:
                    max_dropout = minutes(int(np.max(np.abs(nwp_conf.dropout_timedeltas_minutes))))

                if nwp_conf.max_staleness_minutes is None:
                    max_staleness = None
                else:
                    max_staleness = minutes(nwp_conf.max_staleness_minutes)

                # NWP is a forecast product so gets its own contiguous function
                time_periods = datapipe_copy.find_contiguous_t0_time_periods_nwp(
                    history_duration=minutes(nwp_conf.history_minutes),
                    forecast_duration=minutes(nwp_conf.forecast_minutes),
                    max_staleness=max_staleness,
                    max_dropout=max_dropout,
                    time_dim="init_time_utc",
                )

                contiguous_time_datapipes.append(time_periods)

        else:
            if key == "sat":
                sample_frequency = configuration.input_data.satellite.time_resolution_minutes
                history_duration = configuration.input_data.satellite.history_minutes
                forecast_duration = 0
                time_dim = "time_utc"

            elif key == "hrv":
                sample_frequency = configuration.input_data.hrvsatellite.time_resolution_minutes
                history_duration = configuration.input_data.hrvsatellite.history_minutes
                forecast_duration = 0
                time_dim = "time_utc"

            elif key == "pv":
                sample_frequency = configuration.input_data.pv.time_resolution_minutes
                history_duration = configuration.input_data.pv.history_minutes
                forecast_duration = configuration.input_data.pv.forecast_minutes
                time_dim = "time_utc"

            elif key == "wind":
                sample_frequency = configuration.input_data.wind.time_resolution_minutes
                history_duration = configuration.input_data.wind.history_minutes
                forecast_duration = configuration.input_data.wind.forecast_minutes
                time_dim = "time_utc"

            elif key == "sensor":
                sample_frequency = configuration.input_data.sensor.time_resolution_minutes
                history_duration = configuration.input_data.sensor.history_minutes
                forecast_duration = configuration.input_data.sensor.forecast_minutes
                time_dim = "time_utc"

            elif key == "gsp":
                sample_frequency = configuration.input_data.gsp.time_resolution_minutes
                history_duration = configuration.input_data.gsp.history_minutes
                forecast_duration = configuration.input_data.gsp.forecast_minutes
                time_dim = "time_utc"

            else:
                raise ValueError(f"Unexpected key: {key}")

            datapipes_dict[key], datapipe_copy = datapipes_dict[key].fork(2, buffer_size=5)

            time_periods = datapipe_copy.find_contiguous_t0_time_periods(
                sample_period_duration=minutes(sample_frequency),
                history_duration=minutes(history_duration),
                forecast_duration=minutes(forecast_duration),
                time_dim=time_dim,
            )

            contiguous_time_datapipes.append(time_periods)

    # Find joint overlapping contiguous time periods
    if len(contiguous_time_datapipes) > 1:
        logger.debug("Getting joint time periods")
        overlapping_datapipe = contiguous_time_datapipes[0].filter_to_overlapping_time_periods(
            secondary_datapipes=contiguous_time_datapipes[1:],
        )
    else:
        logger.debug("Skipping getting joint time periods")
        overlapping_datapipe = contiguous_time_datapipes[0]

    # Select time periods and set length
    key_datapipe = key_datapipe.filter_time_periods(time_periods=overlapping_datapipe)

    t0_datapipe = key_datapipe.pick_t0_times()#return_all=True, shuffle=shuffle)


    return t0_datapipe


def construct_time_pipeline(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Tuple[IterDataPipe, IterDataPipe]:
    """Construct time pipeline for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time for time datapipe.
        end_time: Maximum time for time datapipe.
    """

    datapipes_dict = _get_datapipes_dict(
        config_filename,
    )

    # Pull out config file
    config = datapipes_dict.pop("config")

    if (start_time is not None) or (end_time is not None):
        datapipes_dict["gsp"] = datapipes_dict["gsp"].filter_times(start_time, end_time)

    # Get overlapping time periods
    t0_datapipe = create_t0_datapipe(
        datapipes_dict,
        configuration=config,
        shuffle=True,
    )

    return t0_datapipe


def xr_compute(xr_data):
    return xr_data.compute()


@functional_datapipe("select_all_gsp_spatial_slices_pixels")
class SelectAllGSPSpatialSlicePixelsIterDataPipe(IterDataPipe):
    """Select all the spatial slices"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        locations: List[Location],
        roi_height_pixels: int,
        roi_width_pixels: int,
        allow_partial_slice: bool = False,
        location_idx_name: Optional[str] = None,
    ):
        """
        Select spatial slice based off pixels from point of interest

        If `allow_partial_slice` is set to True, then slices may be made which intersect the border
        of the input data. The additional x and y cordinates that would be required for this slice
        are extrapolated based on the average spacing of these coordinates in the input data.
        However, currently slices cannot be made where the centre of the window is outside of the
        input data.

        Args:
            source_datapipe: Datapipe of Xarray data
            roi_height_pixels: ROI height in pixels
            roi_width_pixels: ROI width in pixels
            allow_partial_slice: Whether to allow a partial slice.
            location_idx_name: Name for location index of unstructured grid data,
                None if not relevant
        """
        self.source_datapipe = source_datapipe
        self.locations = locations
        self.roi_height_pixels = roi_height_pixels
        self.roi_width_pixels = roi_width_pixels
        self.allow_partial_slice = allow_partial_slice
        self.location_idx_name = location_idx_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
            
            loc_slices = []
            
            for location in self.locations:
                
                if self.location_idx_name is not None:
                    selected = _get_points_from_unstructured_grids(
                        xr_data=xr_data,
                        location=location,
                        location_idx_name=self.location_idx_name,
                        num_points=self.roi_width_pixels * self.roi_height_pixels,
                    )
                    yield selected

                if xr_coords == "geostationary":
                    center_idx: Location = _get_idx_of_pixel_closest_to_poi_geostationary(
                        xr_data=xr_data,
                        center_osgb=location,
                    )
                else:
                    center_idx: Location = _get_idx_of_pixel_closest_to_poi(
                        xr_data=xr_data,
                        location=location,
                    )

                selected = select_spatial_slice_pixels(
                    xr_data,
                    center_idx,
                    self.roi_width_pixels,
                    self.roi_height_pixels,
                    xr_x_dim,
                    xr_y_dim,
                    allow_partial_slice=self.allow_partial_slice,
                )
                
                loc_slices.append(selected)

            yield loc_slices
            
            
            
@functional_datapipe("select_all_gsp_spatial_slice_meters")
class SelectAllGSPSpatialSliceMetersIterDataPipe(IterDataPipe):
    """Select spatial slice based off meters from point of interest"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        locations: List[Location],
        roi_height_meters: int,
        roi_width_meters: int,
        dim_name: Optional[str] = None,  # "pv_system_id",
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            location_datapipe: Location datapipe
            roi_height_meters: ROI height in meters
            roi_width_meters: ROI width in meters
            dim_name: Dimension name to select for ID, None for coordinates

        Notes:
            Using spatial slicing based on distance rather than number of pixels will often yield
            slices which can vary by 1 pixel in height and/or width.

            E.g. Suppose the Xarray data has x-coords = [1,2,3,4,5]. We want to slice a spatial
            window with a size which equates to 2.2 along the x-axis. If we choose to slice around
            the point x=3 this will slice out the x-coords [2,3,4]. If we choose to slice around the
            point x=2.5 this will slice out the x-coords [2,3]. Hence the returned slice can have
            size either 2 or 3 in the x-axis depending on the spatial location selected.

            Also, if selecting over a large span of latitudes, this may also causes pixel sizes of
            the yielded outputs to change. For example, if the Xarray data is on a regularly spaced
            longitude-latitude grid, then the structure of the grid means that the longitudes near
            to the poles are spaced closer together (measured in meters) than at the equator. So
            slices near the equator will have less pixels in the x-axis than slices taken near the
            poles.
        """
        self.source_datapipe = source_datapipe
        self.locations = locations
        self.roi_height_meters = roi_height_meters
        self.roi_width_meters = roi_width_meters
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            loc_slices = []
            
            for location in self.locations:

                # Get the spatial coords of the xarray data
                xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

                half_height = self.roi_height_meters // 2
                half_width = self.roi_width_meters // 2

                # Find the bounding box values for the location in either lat-lon or OSGB coord systems
                if location.coordinate_system == "lon_lat":
                    right, top = move_lon_lat_by_meters(
                        location.x,
                        location.y,
                        half_width,
                        half_height,
                    )
                    left, bottom = move_lon_lat_by_meters(
                        location.x,
                        location.y,
                        -half_width,
                        -half_height,
                    )

                elif location.coordinate_system == "osgb":
                    left = location.x - half_width
                    right = location.x + half_width
                    bottom = location.y - half_height
                    top = location.y + half_height

                else:
                    raise ValueError(
                        f"Location coord system not recognized: {location.coordinate_system}"
                    )

                # Change the bounding coordinates [left, right, bottom, top] to the same
                # coordinate system as the xarray data
                (left, right), (bottom, top) = convert_coords_to_match_xarray(
                    x=np.array([left, right], dtype=np.float32),
                    y=np.array([bottom, top], dtype=np.float32),
                    from_coords=location.coordinate_system,
                    xr_data=xr_data,
                )

                # Do it off coordinates, not ID
                if self.dim_name is None:
                    # Select a patch from the xarray data
                    x_mask = (left <= xr_data[xr_x_dim]) & (xr_data[xr_x_dim] <= right)
                    y_mask = (bottom <= xr_data[xr_y_dim]) & (xr_data[xr_y_dim] <= top)
                    selected = xr_data.isel({xr_x_dim: x_mask, xr_y_dim: y_mask})

                else:
                    # Select data in the region of interest and ID:
                    # This also works for unstructured grids

                    id_mask = (
                        (left <= xr_data[xr_x_dim])
                        & (xr_data[xr_x_dim] <= right)
                        & (bottom <= xr_data[xr_y_dim])
                        & (xr_data[xr_y_dim] <= top)
                    )
                    selected = xr_data.isel({self.dim_name: id_mask})

                loc_slices.append(selected)
            yield loc_slices


class ConvertWrapper(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        convert_class: Type[IterDataPipe]
    ):
        self.source_datapipe = source_datapipe
        self.convert_class = convert_class

    def __iter__(self):
        for concurrent_samples in self.source_datapipe:
            dp = self.convert_class(IterableWrapper(concurrent_samples))            
            stacked_converted_values = stack_np_examples_into_batch([x for x in iter(dp)])
            yield stacked_converted_values




def construct_sliced_data_pipeline(
    config_filename: str,
    t0_datapipe: IterDataPipe,
    production: bool = False,
    check_satellite_no_zeros: bool = False,
) -> IterDataPipe:
    """Constructs data pipeline for the input data config file.

    This yields samples from the location and time datapipes.

    Args:
        config_filename: Path to config file.
        t0_datapipe: Datapipe yielding times.
        production: Whether constucting pipeline for production inference.
        check_satellite_no_zeros: Whether to check that satellite data has no zeros.
    """

    datapipes_dict = _get_datapipes_dict(
        config_filename,
        production=production,
    )
    
    ds_gsp = next(
        iter(
            open_and_return_datapipes(
                config_filename,
                use_gsp=True,
                use_nwp=False,
                use_pv=False,
                use_sat=False,
                use_hrv=False,
                use_topo=False,
            )["gsp"]
        )
    )
    
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)
    
    locations = [gsp_id_to_loc(gsp_id) for gsp_id in range(1, 318)]


    configuration = datapipes_dict.pop("config")

    # Unpack for convenience
    conf_sat = configuration.input_data.satellite
    conf_nwp = configuration.input_data.nwp

    # Slice all of the datasets by time - this is an in-place operation
    slice_datapipes_by_time(datapipes_dict, t0_datapipe, configuration, production)

    # Spatially slice, normalize, and convert data to numpy arrays
    numpy_modalities = []

    if "nwp" in datapipes_dict:
        nwp_numpy_modalities = dict()

        for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():

            nwp_datapipe = nwp_datapipe.map(xr_compute)

            nwp_datapipe = nwp_datapipe.normalize(
                mean=NWP_MEANS[conf_nwp[nwp_key].nwp_provider],
                std=NWP_STDS[conf_nwp[nwp_key].nwp_provider],
            )
            
            nwp_datapipe = nwp_datapipe.select_all_gsp_spatial_slices_pixels(
                locations,
                roi_height_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_height,
                roi_width_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_width,
            )
            

            nwp_numpy_modalities[nwp_key] = ConvertWrapper(
                nwp_datapipe,
                ConvertNWPToNumpyBatch,
            )

        # Combine the NWPs into NumpyBatch
        nwp_numpy_modalities = MergeNWPNumpyModalities(nwp_numpy_modalities)
        numpy_modalities.append(nwp_numpy_modalities)

    if "sat" in datapipes_dict:
        sat_datapipe = datapipes_dict["sat"]

        sat_datapipe = sat_datapipe.map(xr_compute)
        
        sat_datapipe = sat_datapipe.normalize(mean=RSS_MEAN, std=RSS_STD)
        
        sat_datapipe = sat_datapipe.select_all_gsp_spatial_slices_pixels(
            locations,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
        )
        

        numpy_modalities.append(
            ConvertWrapper(
                sat_datapipe,
                ConvertSatelliteToNumpyBatch,
            )
        )

    if "pv" in datapipes_dict:
        # Recombine PV arrays - see function doc for further explanation
        # No spatial slice for PV since it is always the same
        pv_datapipe = (
            datapipes_dict["pv"]
                .zip_ocf(datapipes_dict["pv_future"])
                .map(concat_xr_time_utc)
        )

        pv_datapipe = pv_datapipe.map(normalize_pv)
        pv_datapipe = pv_datapipe.map(fill_nans_in_pv)
        pv_datapipe = pv_datapipe.map(SampleRepeat(317))
    
        numpy_modalities.append(
            ConvertWrapper(
                pv_datapipe,
                ConvertPVToNumpyBatch,
            )
        )

    # GSP always assumed to be in data
    #location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
    gsp_future_datapipe = datapipes_dict["gsp_future"]
    gsp_future_datapipe = gsp_future_datapipe.select_all_gsp_spatial_slice_meters(
        locations,
        roi_height_meters=1,
        roi_width_meters=1,
        dim_name="gsp_id",
    )

    gsp_datapipe = datapipes_dict["gsp"]
    gsp_datapipe = gsp_datapipe.select_all_gsp_spatial_slice_meters(
        locations,
        roi_height_meters=1,
        roi_width_meters=1,
        dim_name="gsp_id",
    )

    # Recombine GSP arrays - see function doc for further explanation
    gsp_datapipe = (
        gsp_datapipe
        .zip_ocf(gsp_future_datapipe)
        .map(ZipFunction(concat_xr_time_utc))
        .map(SampleFunction(normalize_gsp))
    )
    
    numpy_modalities.append(
        ConvertWrapper(
            gsp_datapipe,
            ConvertGSPToNumpyBatch,
        )
    )

    logger.debug("Combine all the data sources")
    combined_datapipe = MergeNumpyModalities(numpy_modalities).add_sun_position(modality_name="gsp")

    return combined_datapipe


def pvnet_all_gsp_datapipe(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> IterDataPipe:
    """
    Construct pvnet pipeline for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time at which a sample can be selected.
        end_time: Maximum time at which a sample can be selected.
    """
    logger.info("Constructing pvnet pipeline")

    # Open datasets from the config and filter to useable location-time pairs
    t0_datapipe = construct_time_pipeline(
        config_filename,
        start_time,
        end_time,
    )

    # In this function we re-open the datasets to make a clean separation before/after sharding
    # This function
    datapipe = construct_sliced_data_pipeline(
        config_filename,
        t0_datapipe,
    )

    return datapipe


if __name__=="__main__":
    import time
    
    t0 = time.time()
    dp = pvnet_all_gsp_datapipe(
        config_filename="/home/jamesfulton/repos/PVNet/configs/datamodule/configuration/gcp_configuration.yaml"
    )
    
    b  = next(iter(dp))
    print(time.time() - t0)