import numpy as np
import xarray as xr

from ocf_datapipes.select import (
    PickLocations,
    SelectSpatialSliceMeters,
    SelectSpatialSlicePixels,
)
from ocf_datapipes.select.select_spatial_slice import (
    _get_idx_of_pixel_closest_to_poi_geostationary,
    slice_spatial_pixel_window_from_xarray,
)
from ocf_datapipes.utils import Location


def test_slice_spatial_pixel_window_from_xarray_function():
    # Create dummy data
    x = np.arange(100)
    y = np.arange(100)[::-1]

    xr_data = xr.Dataset(
        data_vars=dict(
            data=(["x", "y"], np.random.normal(size=(len(x), len(y)))),
        ),
        coords=dict(
            x=(["x"], x),
            y=(["y"], y),
        ),
    )

    center_idx = Location(x=10, y=10, coordinate_system="idx")

    # Select window which lies within data
    xr_selected = slice_spatial_pixel_window_from_xarray(
        xr_data,
        center_idx,
        width_pixels=10,
        height_pixels=10,
        xr_x_dim="x",
        xr_y_dim="y",
        allow_partial_slice=True,
    )

    assert (xr_selected.x.values == np.arange(5, 15)).all()
    assert (xr_selected.y.values == np.arange(85, 95)[::-1]).all()
    assert not xr_selected.data.isnull().any()

    # Select window where the edge of the window lies at the edge of the data
    xr_selected = slice_spatial_pixel_window_from_xarray(
        xr_data,
        center_idx,
        width_pixels=20,
        height_pixels=20,
        xr_x_dim="x",
        xr_y_dim="y",
        allow_partial_slice=True,
    )

    assert (xr_selected.x.values == np.arange(0, 20)).all()
    assert (xr_selected.y.values == np.arange(80, 100)[::-1]).all()
    assert not xr_selected.data.isnull().any()

    # Select window which is partially outside the boundary of the data
    xr_selected = slice_spatial_pixel_window_from_xarray(
        xr_data,
        center_idx,
        width_pixels=30,
        height_pixels=30,
        xr_x_dim="x",
        xr_y_dim="y",
        allow_partial_slice=True,
    )

    assert (xr_selected.x.values == np.arange(-5, 25)).all(), xr_selected.x.values
    assert (xr_selected.y.values == np.arange(75, 105)[::-1]).all(), xr_selected.y.values
    assert xr_selected.data.isnull().sum() == 275


def test_select_spatial_slice_meters_passiv(passiv_datapipe):
    loc_datapipe = PickLocations(passiv_datapipe)
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
    loc_datapipe = PickLocations(passiv_datapipe)
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
    loc_datapipe = PickLocations(passiv_datapipe)
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
    loc_datapipe = PickLocations(passiv_datapipe, return_all=True)
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
    loc_datapipe = PickLocations(passiv_datapipe)
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
    loc_datapipe = PickLocations(passiv_datapipe, return_all=True)
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

def test_get_idx_of_pixel_closest_to_poi_geostationary_lon_lat_location():
    # Create dummy data
    x = np.arange(5000000, -5000000, -5000)
    y = np.arange(5000000, -5000000, -5000)[::-1]

    xr_data = xr.Dataset(
        data_vars=dict(
            data=(["x_geostationary", "y_geostationary"], np.random.normal(size=(len(x), len(y)))),
        ),
        coords=dict(
            x_geostationary=(["x_geostationary"], x),
            y_geostationary=(["y_geostationary"], y),
        ),
    )
    xr_data.attrs["area"] = 'msg_seviri_iodc_3km:\n  description: MSG SEVIRI Indian Ocean Data Coverage service area definition with\n    3 km resolution\n  projection:\n    proj: geos\n    lon_0: 41.5\n    h: 35785831\n    x_0: 0\n    y_0: 0\n    a: 6378169\n    rf: 295.488065897014\n    no_defs: null\n    type: crs\n  shape:\n    height: 3712\n    width: 3712\n  area_extent:\n    lower_left_xy: [5000000, 5000000]\n    upper_right_xy: [-5000000, -5000000]\n    units: m\n'


    center = Location(x=77.1, y=28.6, coordinate_system="lon_lat")

    location_center_idx = _get_idx_of_pixel_closest_to_poi_geostationary(xr_data=xr_data, center_coordinate=center)

    assert location_center_idx.coordinate_system == 'idx'
    assert location_center_idx.x == 2000
