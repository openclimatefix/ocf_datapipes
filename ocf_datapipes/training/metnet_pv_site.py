"""Create the training/validation datapipe for training the national MetNet/-2 Model"""
import datetime
import logging
from pathlib import Path
from typing import Union

import xarray
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.convert import ConvertPVToNumpy
from ocf_datapipes.select import LocationPicker
from ocf_datapipes.training.common import (
    add_selected_time_slices_from_datapipes,
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
)
from ocf_datapipes.transform.xarray import PreProcessMetNet
from ocf_datapipes.utils.consts import NEW_NWP_MEAN, NEW_NWP_STD, RSS_MEAN, RSS_STD
from ocf_datapipes.utils.future import ThreadPoolMapperIterDataPipe as ThreadPoolMapper

xarray.set_options(keep_attrs=True)
logger = logging.getLogger("metnet_datapipe")
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


def _remove_nans(x):
    return x.fillna(0.0)


def _load_xarray_values(x):
    return x.load()


def metnet_site_datapipe(
    configuration_filename: Union[Path, str],
    use_sun: bool = True,
    use_nwp: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_pv: bool = True,
    use_topo: bool = True,
    output_size: int = 256,
    pv_in_image: bool = False,
    start_time: datetime.datetime = datetime.datetime(2014, 1, 1),
    end_time: datetime.datetime = datetime.datetime(2023, 1, 1),
    center_size_meters: int = 64_000,
    context_size_meters: int = 512_000,
    batch_size: int = 1,
) -> IterDataPipe:
    """
    Make PV data pipe

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
        output_size: Size, in pixels, of the output image
        pv_in_image: Add PV history as channels in MetNet image
        center_size_meters: Center size for MeNet cutouts, in meters
        context_size_meters: Context area size in meters
        batch_size: Batch size for the datapipe

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
    # Load PV data
    used_datapipes["pv"] = (
        used_datapipes["pv"].select_train_test_time(start_time, end_time).pv_interpolate_infill()
    )

    # Now get overlapping time periods
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes, key_for_t0="pv")

    # And now get time slices
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)

    # Now do the extra processing
    pv_history = used_datapipes["pv"].normalize(normalize_fn=normalize_pv)
    pv_datapipe = used_datapipes["pv_future"].normalize(normalize_fn=normalize_pv)
    # Split into PV for target, and one for history
    pv_datapipe, pv_loc_datapipe = pv_datapipe.fork(2)
    pv_loc_datapipe, pv_id_datapipe = LocationPicker(pv_loc_datapipe).fork(2)
    pv_history = pv_history.select_id(pv_id_datapipe, data_source_name="pv")

    if "nwp" in used_datapipes.keys():
        # take nwp time slices
        logger.debug("Take NWP time slices")
        nwp_datapipe = used_datapipes["nwp"].normalize(mean=NEW_NWP_MEAN, std=NEW_NWP_STD)
        pv_loc_datapipe, pv_nwp_image_loc_datapipe = pv_loc_datapipe.fork(2)
        # context_size is the largest it would need
        nwp_datapipe = nwp_datapipe.select_spatial_slice_meters(
            pv_nwp_image_loc_datapipe,
            roi_height_meters=context_size_meters,
            roi_width_meters=context_size_meters,
            dim_name=None,
        )
        # Multithread the data
        nwp_datapipe = ThreadPoolMapper(
            nwp_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
        )

    if "sat" in used_datapipes.keys():
        logger.debug("Take Satellite time slices")
        # take sat time slices
        sat_datapipe = used_datapipes["sat"].normalize(mean=RSS_MEAN, std=RSS_STD)
        pv_loc_datapipe, pv_sat_image_loc_datapipe = pv_loc_datapipe.fork(2)
        sat_datapipe = sat_datapipe.select_spatial_slice_meters(
            pv_sat_image_loc_datapipe,
            roi_height_meters=context_size_meters,
            roi_width_meters=context_size_meters,
            dim_name=None,
        )
        sat_datapipe = ThreadPoolMapper(
            sat_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
        )

    if "hrv" in used_datapipes.keys():
        logger.debug("Take HRV Satellite time slices")
        sat_hrv_datapipe = used_datapipes["hrv"].normalize(mean=RSS_MEAN, std=RSS_STD)
        pv_loc_datapipe, pv_hrv_image_loc_datapipe = pv_loc_datapipe.fork(2)
        sat_hrv_datapipe = sat_hrv_datapipe.select_spatial_slice_meters(
            pv_hrv_image_loc_datapipe,
            roi_height_meters=context_size_meters,
            roi_width_meters=context_size_meters,
            dim_name=None,
        )
        sat_hrv_datapipe = ThreadPoolMapper(
            sat_hrv_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
        )

    if "topo" in used_datapipes.keys():
        topo_datapipe = used_datapipes["topo"].map(_remove_nans)

    # Now combine in the MetNet format
    modalities = []

    if pv_in_image and "hrv" in used_datapipes.keys():
        sat_hrv_datapipe, sat_pv_datapipe = sat_hrv_datapipe.fork(2)
        pv_history = pv_history.create_pv_history_image(image_datapipe=sat_pv_datapipe)
    elif pv_in_image and "sat" in used_datapipes.keys():
        sat_datapipe, sat_pv_datapipe = sat_datapipe.fork(2)
        pv_history = pv_history.create_pv_history_image(image_datapipe=sat_pv_datapipe)
    elif pv_in_image and "nwp" in used_datapipes.keys():
        nwp_datapipe, nwp_pv_datapipe = nwp_datapipe.fork(2)
        pv_history = pv_history.create_pv_history_image(
            image_datapipe=nwp_pv_datapipe, image_dim="osgb"
        )

    if "nwp" in used_datapipes.keys():
        modalities.append(nwp_datapipe)
    if "hrv" in used_datapipes.keys():
        modalities.append(sat_hrv_datapipe)
    if "sat" in used_datapipes.keys():
        modalities.append(sat_datapipe)
    if "topo" in used_datapipes.keys():
        modalities.append(topo_datapipe)
    if pv_in_image:
        modalities.append(pv_history)

    metnet_datapipe = PreProcessMetNet(
        modalities,
        location_datapipe=pv_loc_datapipe,
        center_width=center_size_meters,
        center_height=center_size_meters,  # 64km
        context_height=context_size_meters,
        context_width=context_size_meters,  # 512km
        output_width_pixels=output_size,
        output_height_pixels=output_size,
        add_sun_features=use_sun,
    )

    pv_datapipe = ConvertPVToNumpy(pv_datapipe)

    if not pv_in_image:
        pv_history = pv_history.map(_remove_nans)
        pv_history = ConvertPVToNumpy(pv_history, return_pv_id=True)
        return metnet_datapipe.batch(batch_size).zip_ocf(
            pv_history.batch(batch_size), pv_datapipe.batch(batch_size)
        )
    else:
        return metnet_datapipe.batch(batch_size).zip_ocf(pv_datapipe.batch(batch_size))
