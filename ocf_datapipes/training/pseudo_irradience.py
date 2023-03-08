"""Create the training/validation datapipe for training the national MetNet/-2 Model"""
import datetime
import logging
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.select import LocationPicker
from ocf_datapipes.training.common import (
    add_selected_time_slices_from_datapipes,
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
)
from ocf_datapipes.transform.xarray import StackXarray
from ocf_datapipes.utils.consts import NEW_NWP_MEAN, NEW_NWP_STD, RSS_MEAN, RSS_STD

xarray.set_options(keep_attrs=True)
logger = logging.getLogger("pseudo_irradiance_datapipe")
logger.setLevel(logging.DEBUG)


def normalize_pv(x):  # So it can be pickled
    """
    Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.capacity_watt_power


def _remove_nans(x):
    return x.fillna(0.0)


def _get_numpy_from_xarray(x):
    return x.to_numpy()


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
    start_time: datetime.datetime = datetime.datetime(2014, 1, 1),
    end_time: datetime.datetime = datetime.datetime(2023, 1, 1),
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

    Returns: datapipe
    """

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
    used_datapipes["pv"] = used_datapipes["pv"].select_train_test_time(start_time, end_time)

    # Now get overlapping time periods
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes, key_for_t0="pv")

    # And now get time slices
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)

    # Now do the extra processing
    pv_history = used_datapipes["pv"].normalize(normalize_fn=normalize_pv)
    pv_datapipe = used_datapipes["pv_future"].normalize(normalize_fn=normalize_pv)
    # Split into GSP for target, only national, and one for history
    pv_datapipe, pv_loc_datapipe = pv_datapipe.fork(2)
    pv_loc_datapipe, pv_id_datapipe = LocationPicker(pv_loc_datapipe).fork(2)

    if "nwp" in used_datapipes.keys():
        # take nwp time slices
        logger.debug("Take NWP time slices")
        nwp_datapipe = used_datapipes["nwp"].normalize(mean=NEW_NWP_MEAN, std=NEW_NWP_STD)
        pv_loc_datapipe, pv_nwp_image_loc_datapipe = pv_loc_datapipe.fork(2)
        nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
            pv_nwp_image_loc_datapipe,
            roi_height_pixels=size,
            roi_width_pixels=size,
            x_dim_name="x_osgb",
            y_dim_name="y_osgb",
        )

    if "sat" in used_datapipes.keys():
        logger.debug("Take Satellite time slices")
        # take sat time slices
        sat_datapipe = used_datapipes["sat"].normalize(mean=RSS_MEAN, std=RSS_STD)
        pv_loc_datapipe, pv_sat_image_loc_datapipe = pv_loc_datapipe.fork(2)
        sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
            pv_sat_image_loc_datapipe,
            roi_height_pixels=size,
            roi_width_pixels=size,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
        )

    if "hrv" in used_datapipes.keys():
        logger.debug("Take HRV Satellite time slices")
        sat_hrv_datapipe = used_datapipes["hrv"].normalize(mean=RSS_MEAN, std=RSS_STD)
        pv_loc_datapipe, pv_hrv_image_loc_datapipe = pv_loc_datapipe.fork(2)
        sat_hrv_datapipe = sat_hrv_datapipe.select_spatial_slice_pixels(
            pv_hrv_image_loc_datapipe,
            roi_height_pixels=size,
            roi_width_pixels=size,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
        )

    if "topo" in used_datapipes.keys():
        topo_datapipe = used_datapipes["topo"].map(_remove_nans)
        topo_datapipe = topo_datapipe.select_spatial_slice_pixels(
            pv_hrv_image_loc_datapipe,
            roi_height_pixels=size,
            roi_width_pixels=size,
            x_dim_name="x_osgb",
            y_dim_name="y_osgb",
        )
    # Setting seed in these to keep them the same for creating image and metadata
    if "hrv" in used_datapipes.keys():
        sat_hrv_datapipe, sat_gsp_datapipe, sat_meta_datapipe = sat_hrv_datapipe.fork(3)
        pv_history, pv_meta = pv_history.fork(2)
        pv_history = pv_history.create_pv_image(image_datapipe=sat_gsp_datapipe, seed=1337)
        pv_meta = pv_meta.create_pv_metadata_image(image_datapipe=sat_meta_datapipe, seed=1337)
    elif "sat" in used_datapipes.keys():
        sat_datapipe, sat_gsp_datapipe, sat_meta_datapipe = sat_datapipe.fork(3)
        pv_history, pv_meta = pv_history.fork(2)
        pv_history = pv_history.create_pv_image(image_datapipe=sat_gsp_datapipe, seed=1337)
        pv_meta = pv_meta.create_pv_metadata_image(image_datapipe=sat_meta_datapipe, seed=1337)
    elif "nwp" in used_datapipes.keys():
        nwp_datapipe, nwp_gsp_datapipe, nwp_meta_datapipe = nwp_datapipe.fork(3)
        pv_history, pv_meta = pv_history.fork(2)
        pv_history = pv_history.create_pv_image(
            image_datapipe=nwp_gsp_datapipe, image_dim="osgb", seed=1337
        )
        pv_meta = pv_meta.create_pv_metadata_image(
            image_datapipe=nwp_meta_datapipe, image_dim="osgb", seed=1337
        )

    # Need to have future in image as well
    if "hrv" in used_datapipes.keys():
        sat_hrv_datapipe, sat_future_datapipe = sat_hrv_datapipe.fork(2)
        pv_datapipe = pv_datapipe.create_pv_image(image_datapipe=sat_future_datapipe)
    elif "sat" in used_datapipes.keys():
        sat_datapipe, sat_future_datapipe = sat_datapipe.fork(2)
        pv_datapipe = pv_datapipe.create_pv_image(image_datapipe=sat_future_datapipe)
    elif "nwp" in used_datapipes.keys():
        nwp_datapipe, nwp_future_datapipe = nwp_datapipe.fork(2)
        pv_datapipe = pv_datapipe.create_pv_image(
            image_datapipe=nwp_future_datapipe, image_dim="osgb"
        )

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
        nwp_datapipe, time_image_datapipe = nwp_datapipe.fork(2)
        time_image_datapipe = time_image_datapipe.create_time_image(
            image_dim="osgb", time_dim="target_time_utc"
        )
    elif "hrv" in used_datapipes.keys():
        # Want it at highest resolution possible
        sat_hrv_datapipe, time_image_datapipe = sat_hrv_datapipe.fork(2)
        time_image_datapipe = time_image_datapipe.create_time_image()
    elif "sat" in used_datapipes.keys():
        sat_datapipe, time_image_datapipe = sat_datapipe.fork(2)
        time_image_datapipe = time_image_datapipe.create_time_image()
    else:
        time_image_datapipe = None

    modalities = []
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

    pv_datapipe = pv_datapipe.map(_get_numpy_from_xarray)
    pv_meta = pv_meta.map(_get_numpy_from_xarray)
    return stacked_xarray_inputs.zip_ocf(pv_meta, pv_datapipe)  # Makes (Inputs, Label) tuples
