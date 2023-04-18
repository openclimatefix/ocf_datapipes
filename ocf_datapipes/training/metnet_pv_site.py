"""Create the training/validation datapipe for training the national MetNet/-2 Model"""
import datetime
import logging
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

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
    Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.capacity_watt_power


def _remove_nans(x):
    return x.fillna(0.0)


def _load_xarray_values(x):
    return x.load()

def _drop_pv_ids_in_list(x):
    ids_to_drop = [10426, 10512, 10528, 10548, 10630, 10639, 10837, 11591, 12642, 12846, 12847, 12860, 14577, 14674, 16364, 16474, 17166, 26771, 26772, 26786, 26795, 26818, 26835, 26837, 26866, 26870, 26898, 26911, 26928, 26935, 26944, 26951, 26954, 26955, 26963, 26965, 26978, 26983, 26991, 26994, 27003, 27012, 27014, 27046, 27047, 27053, 27056, 2881, 2915, 3005, 3026, 3248, 3250, 3263, 3324, 3333, 3437, 3489, 3805, 3951, 4065, 5512, 5900, 6011, 6330, 6410, 6433, 6442, 6481, 6493, 6504, 6620, 6633, 6641, 6645, 6656, 6676, 6807, 6880, 6991, 6998, 7076, 7177, 7194, 7234, 7238, 7247, 7255, 7256, 7349, 7390, 7393, 7464, 7487, 7527, 7537, 7557, 7595, 7720, 7759, 7762, 7845, 7906, 7932, 8137, 8591, 8856, 8914, 9101, 9107, 9191, 9760]
    ids_to_sel = []
    for i in ids_to_drop:
        if i in x["pv_system_id"].values:
            ids_to_sel.append(i)
    print(len(ids_to_sel))
    other_ids_to_sel = [58982, 59578, 59917, 60462, 61459, 61517, 64141, 67437, 70440, 70838, 70839, 70870, 77297, 78439, 85491, 86377, 89622, 4541, 5237, 24985, 72039, 22735, 41145, 65197, 24827, 48460, 28568, 56058, 6827, 73765, 28366, 25211, 25131, 34526, 59619, 57162, 20590, 23721, 30352, 4986, 13386, 16223, 23802, 9462, 5037, 6904, 27693, 5185, 5299, 5536, 5660, 6557, 6565, 6711, 7059, 7129, 7603, 7874, 9575, 10095, 10830, 17754, 18889, 19184, 20478, 20724, 20966, 20987, 21156, 21201, 21252, 21858, 21917, 21941, 21971, 21993, 22238, 22867, 23359, 23999, 24026, 24501, 25726, 26189, 27127, 27205, 27368, 27452, 27453, 28156, 28563, 28574, 29366, 29927, 30527, 30765, 31291, 31571, 34650, 35186, 35208, 36265, 37045, 37306, 39085, 46365, 47677, 47756, 48083, 48119, 48382, 54917]
    ids_to_sel2 = []
    for i in other_ids_to_sel:
        if i in x["pv_system_id"].values:
            ids_to_sel2.append(i)
    print(len(ids_to_sel2))
    return x.sel(pv_system_id=ids_to_sel)

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
    match_simon: bool = True,
) -> IterDataPipe:
    """
    Make GSP national data pipe

    Currently only has GSP and NWP's in them

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
        match_simon: Match the sites that Simon tested on

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
    # Now select the sites that Simon tested on
    if match_simon:
        pv_history = pv_history.map(_drop_pv_ids_in_list)
        pv_datapipe = pv_datapipe.map(_drop_pv_ids_in_list)
    # Split into GSP for target, only national, and one for history
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
            x_dim_name="x_osgb",
            y_dim_name="y_osgb",
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
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
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
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
        )
        sat_hrv_datapipe = ThreadPoolMapper(
            sat_hrv_datapipe, _load_xarray_values, max_workers=8, scheduled_tasks=batch_size
        )

    if "topo" in used_datapipes.keys():
        topo_datapipe = used_datapipes["topo"].map(_remove_nans)

    # Now combine in the MetNet format
    modalities = []

    if pv_in_image and "hrv" in used_datapipes.keys():
        sat_hrv_datapipe, sat_gsp_datapipe = sat_hrv_datapipe.fork(2)
        pv_history = pv_history.create_pv_history_image(image_datapipe=sat_gsp_datapipe)
    elif pv_in_image and "sat" in used_datapipes.keys():
        sat_datapipe, sat_gsp_datapipe = sat_datapipe.fork(2)
        pv_history = pv_history.create_pv_history_image(image_datapipe=sat_gsp_datapipe)
    elif pv_in_image and "nwp" in used_datapipes.keys():
        nwp_datapipe, nwp_gsp_datapipe = nwp_datapipe.fork(2)
        pv_history = pv_history.create_pv_history_image(
            image_datapipe=nwp_gsp_datapipe, image_dim="osgb"
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
