"""Create the training/validation datapipe for UK PVNet batches for all GSPs

The main public functions are:

    [1] `pvnet_all_gsp_datapipe()`
        This constructs a datapipe yielding batches with inputs for all 317 UK GSPs for random t0
        times.

    [2] `construct_sliced_data_pipeline()`
        Given a datapipe yielding t0 times, this function constructs a datapipe yielding batches
        with inputs for all 317 UK GSPs for the yielded t0 times. This function is used inside [1].

"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union

import xarray as xr
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.batch import MergeNumpyModalities, MergeNWPNumpyModalities
from ocf_datapipes.batch.merge_numpy_examples_to_batch import stack_np_examples_into_batch
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.convert.numpy_batch import (
    convert_gsp_to_numpy_batch,
    convert_nwp_to_numpy_batch,
    convert_pv_to_numpy_batch,
    convert_satellite_to_numpy_batch,
)
from ocf_datapipes.load.gsp.utils import GSPLocationLookup
from ocf_datapipes.select.select_spatial_slice import (
    select_spatial_slice_meters,
    select_spatial_slice_pixels,
)
from ocf_datapipes.training.common import (
    _get_datapipes_dict,
    check_nans_in_satellite_data,
    concat_xr_time_utc,
    create_valid_t0_periods_datapipe,
    fill_nans_in_arrays,
    fill_nans_in_pv,
    normalize_gsp,
    normalize_pv,
    slice_datapipes_by_time,
)
from ocf_datapipes.utils.consts import (
    NWP_MEANS,
    NWP_STDS,
    RSS_MEAN,
    RSS_STD,
)
from ocf_datapipes.utils.location import Location

xr.set_options(keep_attrs=True)
logger = logging.getLogger("pvnet_all_gsp_datapipe")


# ---------------------------------- Utility datapipes ---------------------------------


def xr_compute(xr_data):
    """Compute the xarray object"""
    return xr_data.compute()


class SampleRepeat:
    """Use a single input element to create a list of identical values"""

    def __init__(self, num_repeats):
        """Use a single input element to create a list of identical values

        Args:
            num_repeats: Length of the returned list of duplicated values
        """
        self.num_repeats = num_repeats

    def __call__(self, x):
        """Repeat the input a number of times as a list"""
        return [x for _ in range(self.num_repeats)]


@functional_datapipe("list_map")
class ListMap(IterDataPipe):
    """Datapipe used to appky function to each item in yielded list"""

    def __init__(self, source_datapipe: IterDataPipe, func, *args, **kwargs):
        """Datapipe used to appky function to each item in yielded list.

        Args:
            source_datapipe: The source datapipe yielding lists of samples
            func: The function to apply to all items in the list
            *args: Args to pass to the function
            **kwargs: Keyword arguments to pass to the function

        """

        self.source_datapipe = source_datapipe
        self.func = func
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        for element_list in self.source_datapipe:
            yield [self.func(x, *self._args, **self._kwargs) for x in element_list]


# ------------------------------ Multi-location datapipes ------------------------------
# These are datapipes rewritten to run on all GSPs


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
        Select spatial slices for all GSPs

        If `allow_partial_slice` is set to True, then slices may be made which intersect the border
        of the input data. The additional x and y cordinates that would be required for this slice
        are extrapolated based on the average spacing of these coordinates in the input data.
        However, currently slices cannot be made where the centre of the window is outside of the
        input data.

        Args:
            source_datapipe: Datapipe of Xarray data
            locations: List of all locations to create samples for
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
            loc_slices = []

            for location in self.locations:
                selected = select_spatial_slice_pixels(
                    xr_data,
                    location,
                    self.roi_width_pixels,
                    self.roi_height_pixels,
                    self.allow_partial_slice,
                    self.location_idx_name,
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
        dim_name: Optional[str] = None,
    ):
        """
        Select spatial slice based off pixels from point of interest

        Args:
            source_datapipe: Datapipe of Xarray data
            locations: List of all locations to create samples for
            roi_width_meters: ROI width in meters
            roi_height_meters: ROI height in meters
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
        self.roi_width_meters = roi_width_meters
        self.roi_height_meters = roi_height_meters
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            loc_slices = []

            for location in self.locations:
                selected = select_spatial_slice_meters(
                    xr_data=xr_data,
                    location=location,
                    roi_width_meters=self.roi_width_meters,
                    roi_height_meters=self.roi_height_meters,
                    dim_name=self.dim_name,
                )

                loc_slices.append(selected)

            yield loc_slices


# ------------------------------- Time pipeline functions ------------------------------


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

    datapipes_dict = _get_datapipes_dict(config_filename)

    # Get config
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
            appropriate t0 times.
        configuration: Configuration object for inputs.
        shuffle: Whether to use the internal shuffle function when yielding times. Else
            location times will be heavily ordered.

    Returns:
        t0 datapipe

    """
    valid_t0_periods_datapipe = create_valid_t0_periods_datapipe(
        datapipes_dict,
        configuration,
        key_for_t0="gsp",
    )

    t0_datapipe = valid_t0_periods_datapipe.pick_t0_times(return_all=True, shuffle=shuffle)

    return t0_datapipe


# ------------------------------- Space pipeline functions -----------------------------


def slice_datapipes_by_space_all_gsps(
    datapipes_dict: dict,
    locations: list[Location],
    configuration: Configuration,
) -> None:
    """Slice the dictionary of datapipes by space in-place"""
    conf_nwp = configuration.input_data.nwp
    conf_sat = configuration.input_data.satellite

    if "nwp" in datapipes_dict:
        for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():
            datapipes_dict["nwp"][nwp_key] = nwp_datapipe.select_all_gsp_spatial_slices_pixels(
                locations,
                roi_width_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_width,
                roi_height_pixels=conf_nwp[nwp_key].nwp_image_size_pixels_height,
            )

    if "sat" in datapipes_dict:
        datapipes_dict["sat"] = datapipes_dict["sat"].select_all_gsp_spatial_slices_pixels(
            locations,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
        )

    if "pv" in datapipes_dict:
        # No spatial slice for PV since it is always the same, just repeat for GSPs
        datapipes_dict["pv"] = datapipes_dict["pv"].map(SampleRepeat(len(locations)))

    # GSP always assumed to be in data
    datapipes_dict["gsp"] = datapipes_dict["gsp"].select_all_gsp_spatial_slice_meters(
        locations,
        roi_width_meters=1,
        roi_height_meters=1,
        dim_name="gsp_id",
    )


# -------------------------------- Processing functions --------------------------------


def pre_spatial_slice_process(datapipes_dict, configuration):
    """Apply pre-processing steps to the dictionary of datapipes in place

    These steps are normalisation and recombining past and future GSP/PV data
    """
    conf_nwp = configuration.input_data.nwp

    if "nwp" in datapipes_dict:
        for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():
            datapipes_dict["nwp"][nwp_key] = nwp_datapipe.map(xr_compute).normalize(
                mean=NWP_MEANS[conf_nwp[nwp_key].nwp_provider],
                std=NWP_STDS[conf_nwp[nwp_key].nwp_provider],
            )

    if "sat" in datapipes_dict:
        datapipes_dict["sat"] = (
            datapipes_dict["sat"].map(xr_compute).normalize(mean=RSS_MEAN, std=RSS_STD)
        )

    if "pv" in datapipes_dict:
        # Recombine PV arrays - see function doc for further explanation
        datapipes_dict["pv"] = (
            datapipes_dict["pv"]
            .zip_ocf(datapipes_dict["pv_future"])
            .map(concat_xr_time_utc)
            .map(normalize_pv)
            .map(fill_nans_in_pv)
        )

        del datapipes_dict["pv_future"]

    # GSP always assumed to be in data
    # Recombine GSP arrays - see function doc for further explanation
    datapipes_dict["gsp"] = (
        datapipes_dict["gsp"]
        .zip_ocf(datapipes_dict["gsp_future"])
        .map(concat_xr_time_utc)
        .map(normalize_gsp)
    )

    del datapipes_dict["gsp_future"]


def post_spatial_slice_process(datapipes_dict, check_satellite_no_nans=False):
    """Convert the dictionary of datapipes to NumpyBatches, combine, and fill nans"""

    numpy_modalities = []

    if "nwp" in datapipes_dict:
        nwp_numpy_modalities = dict()

        for nwp_key, nwp_datapipe in datapipes_dict["nwp"].items():
            nwp_numpy_modalities[nwp_key] = nwp_datapipe.list_map(convert_nwp_to_numpy_batch).map(
                stack_np_examples_into_batch
            )

        # Combine the NWPs into NumpyBatch
        nwp_numpy_modalities = MergeNWPNumpyModalities(nwp_numpy_modalities)
        numpy_modalities.append(nwp_numpy_modalities)

    if "sat" in datapipes_dict:
        numpy_modalities.append(
            datapipes_dict["sat"]
            .list_map(convert_satellite_to_numpy_batch)
            .map(stack_np_examples_into_batch)
        )

    if "pv" in datapipes_dict:
        numpy_modalities.append(
            datapipes_dict["pv"]
            .list_map(convert_pv_to_numpy_batch)
            .map(stack_np_examples_into_batch)
        )

    # GSP always assumed to be in data
    numpy_modalities.append(
        datapipes_dict["gsp"].list_map(convert_gsp_to_numpy_batch).map(stack_np_examples_into_batch)
    )

    # Combine all the data sources
    combined_datapipe = MergeNumpyModalities(numpy_modalities).add_sun_position(modality_name="gsp")

    if check_satellite_no_nans:
        # in production we don't want any nans in the satellite data
        combined_datapipe = combined_datapipe.map(check_nans_in_satellite_data)

    combined_datapipe = combined_datapipe.map(fill_nans_in_arrays)

    return combined_datapipe


# --------------------------- High level pipeline functions ----------------------------


def construct_sliced_data_pipeline(
    config_filename: str,
    t0_datapipe: IterDataPipe,
    production: bool = False,
    check_satellite_no_nans: bool = False,
) -> IterDataPipe:
    """Constructs data pipeline for the input data config file.

    This yields samples from the location and time datapipes.

    Args:
        config_filename: Path to config file.
        t0_datapipe: Datapipe yielding times.
        production: Whether constucting pipeline for production inference.
        check_satellite_no_nans: Whether to check that satellite data has no nans.
    """

    datapipes_dict = _get_datapipes_dict(
        config_filename,
        production=production,
    )

    # Get the location objects for all 317 regional GSPs
    gsp_id_to_loc = GSPLocationLookup()
    locations = [gsp_id_to_loc(gsp_id) for gsp_id in range(1, 318)]

    # Pop config
    configuration = datapipes_dict.pop("config")

    # Slice all of the datasets by time - this is an in-place operation
    slice_datapipes_by_time(datapipes_dict, t0_datapipe, configuration, production)

    # Run compute and normalise all the data
    pre_spatial_slice_process(datapipes_dict, configuration)

    # Slice all of the datasets by space - this is an in-place operation
    slice_datapipes_by_space_all_gsps(datapipes_dict, locations, configuration)

    # Convert to NumpyBatch
    combined_datapipe = post_spatial_slice_process(datapipes_dict)

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

    # Open datasets from the config and filter to useable times
    t0_datapipe = construct_time_pipeline(
        config_filename,
        start_time,
        end_time,
    )

    # Shard after we have the times. These are already shuffled so no need to shuffle again
    t0_datapipe = t0_datapipe.sharding_filter()

    # In this function we re-open the datasets to make a clean separation before/after sharding
    # This function
    datapipe = construct_sliced_data_pipeline(
        config_filename,
        t0_datapipe,
    )

    return datapipe
