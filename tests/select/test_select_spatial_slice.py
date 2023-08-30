from ocf_datapipes.select import (
    LocationPicker,
    SelectSpatialSliceMeters,
    SelectSpatialSlicePixels,
)


def test_select_spatial_slice_meters_passiv(passiv_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe)
    passiv_datapipe = SelectSpatialSliceMeters(
        passiv_datapipe,
        location_datapipe=loc_datapipe,
        roi_width_meters=96_000,
        roi_height_meters=96_000,
        dim_name="pv_system_id",
    )
    data = next(iter(passiv_datapipe))
    assert len(data.pv_system_id) == 1


def test_select_spatial_slice_pixels_hrv(passiv_datapipe, sat_hrv_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe)
    sat_hrv_datapipe = SelectSpatialSlicePixels(
        sat_hrv_datapipe,
        location_datapipe=loc_datapipe,
        roi_width_pixels=256,
        roi_height_pixels=128,
    )
    data = next(iter(sat_hrv_datapipe))
    assert len(data.x_geostationary) == 256
    assert len(data.y_geostationary) == 128


def test_select_spatial_slice_pixel_icon_eu(passiv_datapipe, icon_eu_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe)
    icon_eu_datapipe = SelectSpatialSlicePixels(
        icon_eu_datapipe,
        location_datapipe=loc_datapipe,
        roi_width_pixels=256,
        roi_height_pixels=128,
    )
    data = next(iter(icon_eu_datapipe))
    assert len(data.longitude) == 256
    assert len(data.latitude) == 128


def test_select_spatial_slice_pixel_icon_global(passiv_datapipe, icon_global_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe, return_all_locations=True)
    icon_global_datapipe = SelectSpatialSlicePixels(
        icon_global_datapipe,
        location_datapipe=loc_datapipe,
        roi_width_pixels=256,
        roi_height_pixels=128,
        location_idx_name="values",
    )
    data = next(iter(icon_global_datapipe))
    assert len(data.longitude) == 32768
    assert len(data.latitude) == 32768


def test_select_spatial_slice_meters_icon_eu(passiv_datapipe, icon_eu_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe)
    icon_eu_datapipe = SelectSpatialSliceMeters(
        icon_eu_datapipe,
        location_datapipe=loc_datapipe,
        roi_width_meters=70_000,
        roi_height_meters=70_000,
        dim_name=None,
    )
    data = next(iter(icon_eu_datapipe))

    # Grid longitude spacing is 0.0625 degrees which is 4km at latitude 55 degrees North
    # Slice can cover 17 or 18 grid points depending on where its centred
    assert len(data.longitude) in [17, 18]
    # Grid latitude spacing is 0.0625 degrees - around 7km
    assert len(data.latitude) == 10


def test_select_spatial_slice_meters_icon_global(passiv_datapipe, icon_global_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe, return_all_locations=True)
    icon_global_datapipe = SelectSpatialSliceMeters(
        icon_global_datapipe,
        location_datapipe=loc_datapipe,
        roi_width_meters=96_000,
        roi_height_meters=96_000,
        dim_name="values",
    )
    data = next(iter(icon_global_datapipe))
    # ICON global has roughly 13km spacing, so this should be around 7x7 grid
    assert len(data.longitude) == 49
    assert len(data.latitude) == 49
