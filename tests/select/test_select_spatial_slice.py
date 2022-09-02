from ocf_datapipes.select import LocationPicker, SelectSpatialSliceMeters, SelectSpatialSlicePixels


def test_select_spatial_slice_meters_passiv(passiv_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe)
    passiv_datapipe = SelectSpatialSliceMeters(
        passiv_datapipe, location_datapipe=loc_datapipe, roi_width_meters=96_000, roi_height_meters=96_000
    )
    data = next(iter(passiv_datapipe))
    assert len(data.pv_system_id) == 1


def test_select_spatial_slice_pixels_hrv(sat_hrv_datapipe, passiv_datapipe):
    loc_datapipe = LocationPicker(passiv_datapipe)
    sat_hrv_datapipe = SelectSpatialSlicePixels(
        sat_hrv_datapipe,
        location_datapipe=loc_datapipe,
        roi_width_pixels=256,
        roi_height_pixels=128,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    )
    data = next(iter(sat_hrv_datapipe))
    assert len(data.x_geostationary) == 256
    assert len(data.y_geostationary) == 128
