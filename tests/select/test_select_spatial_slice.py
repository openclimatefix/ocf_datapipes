from ocf_datapipes.select import LocationPicker, SelectSpatialSliceMeters, SelectSpatialSlicePixels


def test_select_spatial_slice_meters_passiv(passiv_dp):
    loc_dp = LocationPicker(passiv_dp)
    passiv_dp = SelectSpatialSliceMeters(
        passiv_dp, location_datapipe=loc_dp, roi_width_meters=96_000, roi_height_meters=96_000
    )
    data = next(iter(passiv_dp))
    assert data is not None


def test_select_spatial_slice_pixels_hrv(sat_hrv_dp, passiv_dp):
    loc_dp = LocationPicker(passiv_dp)
    sat_hrv_dp = SelectSpatialSlicePixels(
        sat_hrv_dp,
        location_datapipe=loc_dp,
        roi_width_pixels=256,
        roi_height_pixels=128,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    )
    data = next(iter(sat_hrv_dp))
    assert len(data.x_geostationary) == 256
    assert len(data.y_geostationary) == 128
