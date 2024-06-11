"""Create the training/validation datapipe for training the national MetNet/-2 Model"""

import datetime
import logging
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pvlib
import xarray
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.convert import StackXarray
from ocf_datapipes.select import PickLocations
from ocf_datapipes.training.common import (
    add_selected_time_slices_from_datapipes,
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
)
from ocf_datapipes.utils.consts import UKV_MAX, UKV_MIN
from ocf_datapipes.utils.future import ThreadPoolMapperIterDataPipe as ThreadPoolMapper

xarray.set_options(keep_attrs=True)
logger = logging.getLogger("pseudo_irradiance_datapipe")
logger.setLevel(logging.DEBUG)


def normalize_pv(x):  # So it can be pickled
    """
    Normalize the PV data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.observed_capacity_wp


def _select_non_nan_times(x):
    return x.fillna(0.0)


def _get_numpy_from_xarray(x):
    return x.to_numpy()


def _load_xarray_values(x):
    return x.load()


def _normalize_nwp(x):
    return (x - UKV_MIN) / UKV_MAX


def _resample_to_pixel_size(xr_data, height_pixels, width_pixels) -> np.ndarray:
    if "x_geostationary" in xr_data.dims:
        x_coords = xr_data["x_geostationary"].values
        y_coords = xr_data["y_geostationary"].values
    elif "x_osgb" in xr_data.dims:
        x_coords = xr_data["x_osgb"].values
        y_coords = xr_data["y_osgb"].values
    else:
        x_coords = xr_data["x"].values
        y_coords = xr_data["y"].values
    # Resample down to the number of pixels wanted
    x_coords = np.linspace(x_coords[0], x_coords[-1], num=width_pixels)
    y_coords = np.linspace(y_coords[0], y_coords[-1], num=height_pixels)
    if "x_geostationary" in xr_data.dims:
        xr_data = xr_data.interp(
            x_geostationary=x_coords, y_geostationary=y_coords, method="linear"
        )
    elif "x_osgb" in xr_data.dims:
        xr_data = xr_data.interp(x_osgb=x_coords, y_osgb=y_coords, method="linear")
    else:
        xr_data = xr_data.interp(x=x_coords, y=y_coords, method="linear")
    # Extract just the data now
    return xr_data.load()


def _normalize_by_pvlib(pv_system):
    """
    Normalize the output by pv_libs poa_global

    Args:
        pv_system: PV System in Xarray DataArray

    Returns:
        PV System in xarray DataArray, but normalized values
    """
    # TODO Add elevation
    pvlib_loc = pvlib.location.Location(
        latitude=pv_system.latitude.values, longitude=pv_system.longitude.values
    )
    times = pd.DatetimeIndex(pv_system.time_utc.values)
    solar_position = pvlib_loc.get_solarposition(times=times)
    clear_sky = pvlib_loc.get_clearsky(times)
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        pv_system.tilt.values,
        pv_system.orientation.values,
        solar_zenith=solar_position["zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        dhi=clear_sky["dhi"],
        ghi=clear_sky["ghi"],
    )
    # Guess want fraction of total irradiance on panel, to get fraction to do with capacity
    fraction_clear_sky = total_irradiance["poa_global"] / (
        clear_sky["dni"] + clear_sky["dhi"] + clear_sky["ghi"]
    )
    print(fraction_clear_sky)
    pv_system /= pv_system.observed_capacity_wp
    print(pv_system)
    pv_system *= fraction_clear_sky
    print(pv_system)
    print("---------------------------------------------------")
    return pv_system


def _get_meta(xr_data):
    tilt = xr_data["tilt"].values
    orientation = xr_data["orientation"].values
    combined = np.array([tilt, orientation])
    return combined


def _get_values(xr_data):
    xr_data = normalize_pv(xr_data)
    return xr_data.values


def _filter_tilt_orientation(xr_data):
    xr_data = xr_data.where(np.isfinite(xr_data.orientation) & np.isfinite(xr_data.tilt), drop=True)
    return xr_data


def _keep_pv_ids_in_list(x):
    ids_to_sel = [
        10426,
        10512,
        10528,
        10548,
        10630,
        10639,
        10837,
        11591,
        12642,
        12846,
        12847,
        12860,
        14577,
        14674,
        16364,
        16474,
        17166,
        26771,
        26772,
        26786,
        26795,
        26818,
        26835,
        26837,
        26866,
        26870,
        26898,
        26911,
        26928,
        26935,
        26944,
        26951,
        26954,
        26955,
        26963,
        26965,
        26978,
        26983,
        26991,
        26994,
        27003,
        27012,
        27014,
        27046,
        27047,
        27053,
        27056,
        2881,
        2915,
        3005,
        3026,
        3248,
        3250,
        3263,
        3324,
        3333,
        3437,
        3489,
        3805,
        3951,
        4065,
        5512,
        5900,
        6011,
        6330,
        6410,
        6433,
        6442,
        6481,
        6493,
        6504,
        6620,
        6633,
        6641,
        6645,
        6656,
        6676,
        6807,
        6880,
        6991,
        6998,
        7076,
        7177,
        7194,
        7234,
        7238,
        7247,
        7255,
        7256,
        7349,
        7390,
        7393,
        7464,
        7487,
        7527,
        7537,
        7557,
        7595,
        7720,
        7762,
        7845,
        7906,
        7932,
        8137,
        8591,
        8856,
        8914,
        9101,
        9107,
        9191,
        9760,
    ]
    return x.sel(pv_system_id=ids_to_sel)


def _drop_pv_ids_in_list(x):
    ids_to_sel = [
        10426,
        10512,
        10528,
        10548,
        10630,
        10639,
        10837,
        11591,
        12642,
        12846,
        12847,
        12860,
        14577,
        14674,
        16364,
        16474,
        17166,
        26771,
        26772,
        26786,
        26795,
        26818,
        26835,
        26837,
        26866,
        26870,
        26898,
        26911,
        26928,
        26935,
        26944,
        26951,
        26954,
        26955,
        26963,
        26965,
        26978,
        26983,
        26991,
        26994,
        27003,
        27012,
        27014,
        27046,
        27047,
        27053,
        27056,
        2881,
        2915,
        3005,
        3026,
        3248,
        3250,
        3263,
        3324,
        3333,
        3437,
        3489,
        3805,
        3951,
        4065,
        5512,
        5900,
        6011,
        6330,
        6410,
        6433,
        6442,
        6481,
        6493,
        6504,
        6620,
        6633,
        6641,
        6645,
        6656,
        6676,
        6807,
        6880,
        6991,
        6998,
        7076,
        7177,
        7194,
        7234,
        7238,
        7247,
        7255,
        7256,
        7349,
        7390,
        7393,
        7464,
        7487,
        7527,
        7537,
        7557,
        7595,
        7720,
        7762,
        7845,
        7906,
        7932,
        8137,
        8591,
        8856,
        8914,
        9101,
        9107,
        9191,
        9760,
    ]
    return x.sel(pv_system_id=ids_to_sel, drop=True)


def _extract_test_info(x):
    # Concatenate the time and pv_id into a 1D numpy array
    return x.time_utc.values


def _get_id_from_location(x):
    return x.id


def pseudo_irradiance_datapipe(
    configuration_filename: Union[Path, str],
    use_sun: bool = True,
    use_nwp: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_pv: bool = True,
    use_topo: bool = True,
    use_future: bool = False,
    size: int = 256,
    size_meters: int = 256_000,
    use_meters: bool = False,
    start_time: datetime.datetime = datetime.datetime(2014, 1, 1),
    end_time: datetime.datetime = datetime.datetime(2023, 1, 1),
    batch_size: int = 1,
    normalize_by_pvlib: bool = True,
    one_d: bool = False,
    is_test: bool = False,
) -> IterDataPipe:
    """
    Make Pseudo-Irradience Datapipe

    This outputs PV generation

    Args:
        configuration_filename: the configruation filename for the pipe
        use_sun: Whether to add sun features or not
        use_pv: Whether to use PV input or not
        use_hrv: Whether to use HRV Satellite or not
        use_sat: Whether to use non-HRV Satellite or not
        use_nwp: Whether to use NWP or not
        use_topo: Whether to use topographic map or not
        start_time: Start time to select on
        end_time: End time to select from
        use_future: Whether to use future frames as well (for labelling)
        size: Size, in pixels, of the output image
        batch_size: Batch size for the datapipe
        one_d: Whether to return a 1D array or not, i.e. a single PV site in the center as
            opposed to a 2D array of PV sites
        size_meters: Size, in meters, of the output image
        use_meters: Whether to use meters or pixels
        normalize_by_pvlib: Whether to normalize the PV generation by the PVLib generation
        is_test: Whether to return the test set or not

    Returns: datapipe
    """

    # Use partial to define the _resample_to_pixel_size function using size
    resample_to_pixel_size = partial(_resample_to_pixel_size, height_pixels=size, width_pixels=size)
    # load datasets
    used_datapipes = open_and_return_datapipes(
        configuration_filename=configuration_filename,
        use_nwp=use_nwp,
        use_topo=use_topo,
        use_sat=use_sat,
        use_hrv=use_hrv,
        use_gsp=False,
        use_pv=use_pv,
    )
    # Load GSP national data
    used_datapipes["pv"] = used_datapipes["pv"].filter_times(start_time, end_time)

    # Now get overlapping time periods
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(
        used_datapipes,
        key_for_t0="pv",
    )
    # And now get time slices
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)
    # print(used_datapipes.keys())

    # Now do the extra processing
    pv_history = used_datapipes["pv"].map(
        _filter_tilt_orientation
    )  # .normalize(normalize_fn=normalize_pv)
    pv_datapipe = used_datapipes["pv_future"].map(
        _filter_tilt_orientation
    )  # .normalize(normalize_fn=normalize_pv)
    # return pv_datapipe.zip_ocf(pv_history,used_datapipes["sat"])
    if is_test:
        pv_history = pv_history.map(_keep_pv_ids_in_list)
        pv_datapipe = pv_datapipe.map(_keep_pv_ids_in_list)
    else:
        pv_history = pv_history.map(_drop_pv_ids_in_list)
        pv_datapipe = pv_datapipe.map(_drop_pv_ids_in_list)
    # Split into GSP for target, only national, and one for history
    pv_datapipe, pv_loc_datapipe, pv_meta_save = pv_datapipe.fork(3)
    pv_loc_datapipe, pv_sav_loc = PickLocations(
        pv_loc_datapipe,
        return_all_locations=True if is_test else False,
    ).fork(2, buffer_size=-1)
    pv_sav_loc = pv_sav_loc.map(_get_id_from_location)
    pv_meta_save = pv_meta_save.map(_extract_test_info)
    #
    # Select systems here
    if use_meters:
        pv_loc_datapipe, pv_loc_datapipe1, pv_loc_datapipe2 = pv_loc_datapipe.fork(
            3, buffer_size=-1
        )
        pv_datapipe = pv_datapipe.select_spatial_slice_meters(
            pv_loc_datapipe1, roi_height_meters=size_meters, roi_width_meters=size_meters
        )
        pv_history = pv_history.select_spatial_slice_meters(
            pv_loc_datapipe2, roi_height_meters=size_meters, roi_width_meters=size_meters
        )

    if one_d:
        pv_loc_datapipe, pv_one_d_datapipe, pv_one_d_datapipe2 = pv_loc_datapipe.fork(
            3, buffer_size=-1
        )
        pv_datapipe = pv_datapipe.select_id(pv_one_d_datapipe, data_source_name="pv")
        pv_history = pv_history.select_id(pv_one_d_datapipe2, data_source_name="pv")

    if "nwp" in used_datapipes.keys():
        # take nwp time slices
        logger.debug("Take NWP time slices")
        nwp_datapipe = used_datapipes["nwp"].map(_normalize_nwp)
        pv_loc_datapipe, pv_nwp_image_loc_datapipe = pv_loc_datapipe.fork(2)
        if use_meters:
            nwp_datapipe = nwp_datapipe.select_spatial_slice_meters(
                pv_nwp_image_loc_datapipe,
                roi_height_meters=size_meters,
                roi_width_meters=size_meters,
                dim_name=None,
            )
            # nwp_datapipe = nwp_datapipe.map(resample_to_pixel_size)
            nwp_datapipe = ThreadPoolMapper(
                nwp_datapipe, resample_to_pixel_size, max_workers=8, scheduled_tasks=batch_size
            )
        else:
            nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
                pv_nwp_image_loc_datapipe,
                roi_height_pixels=size,
                roi_width_pixels=size,
            )
            # nwp_datapipe = nwp_datapipe.map(_load_xarray_values)
            nwp_datapipe = ThreadPoolMapper(
                nwp_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
            )

    if "sat" in used_datapipes.keys():
        logger.debug("Take Satellite time slices")
        # take sat time slices
        sat_datapipe = used_datapipes["sat"]  # .normalize(mean=RSS_MEAN, std=RSS_STD)
        pv_loc_datapipe, pv_sat_image_loc_datapipe = pv_loc_datapipe.fork(2)
        if use_meters:
            sat_datapipe = sat_datapipe.select_spatial_slice_meters(
                pv_sat_image_loc_datapipe,
                roi_height_meters=size_meters,
                roi_width_meters=size_meters,
                dim_name=None,
            )
            sat_datapipe = ThreadPoolMapper(
                sat_datapipe, resample_to_pixel_size, max_workers=8, scheduled_tasks=batch_size
            )
        else:
            sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
                pv_sat_image_loc_datapipe,
                roi_height_pixels=size,
                roi_width_pixels=size,
            )
            sat_datapipe = ThreadPoolMapper(
                sat_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
            )
    if "hrv" in used_datapipes.keys():
        logger.debug("Take HRV Satellite time slices")
        sat_hrv_datapipe = used_datapipes["hrv"]  # .normalize(mean=RSS_MEAN, std=RSS_STD)
        pv_loc_datapipe, pv_hrv_image_loc_datapipe = pv_loc_datapipe.fork(2)
        if use_meters:
            sat_hrv_datapipe = sat_hrv_datapipe.select_spatial_slice_meters(
                pv_hrv_image_loc_datapipe,
                roi_height_meters=size_meters,
                roi_width_meters=size_meters,
                dim_name=None,
            )
            sat_hrv_datapipe = ThreadPoolMapper(
                sat_hrv_datapipe, resample_to_pixel_size, max_workers=8, scheduled_tasks=batch_size
            )
        else:
            sat_hrv_datapipe = sat_hrv_datapipe.select_spatial_slice_pixels(
                pv_hrv_image_loc_datapipe,
                roi_height_pixels=size,
                roi_width_pixels=size,
            )
            sat_hrv_datapipe = ThreadPoolMapper(
                sat_hrv_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
            )

    if "topo" in used_datapipes.keys():
        topo_datapipe = used_datapipes["topo"].normalize().map(_select_non_nan_times)
        pv_loc_datapipe, pv_hrv_image_loc_datapipe = pv_loc_datapipe.fork(2)
        if use_meters:
            topo_datapipe = topo_datapipe.select_spatial_slice_meters(
                pv_hrv_image_loc_datapipe,
                roi_height_meters=size_meters,
                roi_width_meters=size_meters,
                dim_name=None,
            )
            topo_datapipe = ThreadPoolMapper(
                topo_datapipe, resample_to_pixel_size, max_workers=8, scheduled_tasks=batch_size
            )
        else:
            topo_datapipe = topo_datapipe.select_spatial_slice_pixels(
                pv_hrv_image_loc_datapipe,
                roi_height_pixels=size,
                roi_width_pixels=size,
            )
            topo_datapipe = ThreadPoolMapper(
                topo_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
            )
    # Setting seed in these to keep them the same for creating image and metadata
    if one_d:
        pv_datapipe, pv_meta = pv_datapipe.fork(2)
        pv_meta = pv_meta.map(_get_meta)
        pv_datapipe = pv_datapipe.map(_get_values)
    else:
        if "hrv" in used_datapipes.keys():
            sat_hrv_datapipe, sat_gsp_datapipe = sat_hrv_datapipe.fork(2)
            pv_history, pv_meta = pv_history.create_pv_image(
                image_datapipe=sat_gsp_datapipe,
                make_meta_image=True,
                normalize_by_pvlib=normalize_by_pvlib,
                normalize=not normalize_by_pvlib,
                take_n_pv_values_per_pixel=-1,
            ).unzip(sequence_length=2)
        elif "sat" in used_datapipes.keys():
            sat_datapipe, sat_gsp_datapipe = sat_datapipe.fork(2)
            pv_history, pv_meta = pv_history.create_pv_image(
                image_datapipe=sat_gsp_datapipe,
                make_meta_image=True,
                normalize_by_pvlib=normalize_by_pvlib,
                normalize=not normalize_by_pvlib,
                take_n_pv_values_per_pixel=-1,
            ).unzip(sequence_length=2)
        elif "nwp" in used_datapipes.keys():
            nwp_datapipe, nwp_gsp_datapipe = nwp_datapipe.fork(2)
            pv_history, pv_meta = pv_history.create_pv_image(
                image_datapipe=nwp_gsp_datapipe,
                image_dim="osgb",
                make_meta_image=True,
                normalize_by_pvlib=normalize_by_pvlib,
                normalize=not normalize_by_pvlib,
                take_n_pv_values_per_pixel=-1,
            ).unzip(sequence_length=2)

        # Need to have future in image as well
        if "hrv" in used_datapipes.keys():
            sat_hrv_datapipe, sat_future_datapipe = sat_hrv_datapipe.fork(2)
            pv_datapipe = pv_datapipe.create_pv_image(
                image_datapipe=sat_future_datapipe,
                normalize_by_pvlib=normalize_by_pvlib,
                normalize=not normalize_by_pvlib,
            )
        elif "sat" in used_datapipes.keys():
            sat_datapipe, sat_future_datapipe = sat_datapipe.fork(2)
            pv_datapipe = pv_datapipe.create_pv_image(
                image_datapipe=sat_future_datapipe,
                normalize_by_pvlib=normalize_by_pvlib,
                normalize=not normalize_by_pvlib,
            )
        elif "nwp" in used_datapipes.keys():
            nwp_datapipe, nwp_future_datapipe = nwp_datapipe.fork(2)
            pv_datapipe = pv_datapipe.create_pv_image(
                image_datapipe=nwp_future_datapipe,
                image_dim="osgb",
                normalize_by_pvlib=normalize_by_pvlib,
                normalize=not normalize_by_pvlib,
            )
        pv_datapipe = pv_datapipe.map(_get_numpy_from_xarray)
        pv_meta = pv_meta.map(_get_numpy_from_xarray)

    if use_sun:
        if "nwp" in used_datapipes.keys():
            nwp_datapipe, sun_image_datapipe = nwp_datapipe.fork(2)
            sun_image_datapipe = sun_image_datapipe.create_sun_image(
                normalize=True, image_dim="osgb", time_dim="target_time_utc"
            )
        elif "hrv" in used_datapipes.keys():
            # Want it at highest resolution possible
            sat_hrv_datapipe, sun_image_datapipe = sat_hrv_datapipe.fork(2)
            sun_image_datapipe = sun_image_datapipe.create_sun_image(normalize=True)
        elif "sat" in used_datapipes.keys():
            sat_datapipe, sun_image_datapipe = sat_datapipe.fork(2)
            sun_image_datapipe = sun_image_datapipe.create_sun_image(normalize=True)
    if "nwp" in used_datapipes.keys():
        nwp_datapipe, time_image_datapipe = nwp_datapipe.fork(2, buffer_size=100)
        time_image_datapipe = time_image_datapipe.create_time_image(
            image_dim="osgb", time_dim="target_time_utc"
        )
    elif "hrv" in used_datapipes.keys():
        # Want it at highest resolution possible
        sat_hrv_datapipe, time_image_datapipe = sat_hrv_datapipe.fork(2, buffer_size=100)
        time_image_datapipe = time_image_datapipe.create_time_image()
    elif "sat" in used_datapipes.keys():
        sat_datapipe, time_image_datapipe = sat_datapipe.fork(2, buffer_size=100)
        time_image_datapipe = time_image_datapipe.create_time_image()
    else:
        time_image_datapipe = None

    modalities = []
    if not one_d:
        modalities.append(pv_history)
    if "nwp" in used_datapipes.keys():
        modalities.append(nwp_datapipe)
    if "hrv" in used_datapipes.keys():
        modalities.append(sat_hrv_datapipe)
    if "sat" in used_datapipes.keys():
        modalities.append(sat_datapipe)
    if "topo" in used_datapipes.keys():
        modalities.append(topo_datapipe)
    if use_sun:
        modalities.append(sun_image_datapipe)
    if time_image_datapipe is not None:
        modalities.append(time_image_datapipe)

    stacked_xarray_inputs = StackXarray(modalities)
    return stacked_xarray_inputs.batch(batch_size).zip_ocf(
        pv_meta.batch(batch_size),
        pv_datapipe.batch(batch_size),
        pv_meta_save.batch(batch_size),
        pv_sav_loc.batch(batch_size),
        pv_history.batch(batch_size),
        pv_loc_datapipe.batch(batch_size),
    )  # Makes (Inputs, Label) tuples
