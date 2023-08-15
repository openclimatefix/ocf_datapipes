from ocf_datapipes.transform.xarray import ConvertPressureLevelsToSeparateVariables
import xarray as xr


def test_convert_pressure_levels_to_separate_variables(icon_eu_datapipe):
    icon_eu_datapipe = ConvertPressureLevelsToSeparateVariables(icon_eu_datapipe)
    data = next(iter(icon_eu_datapipe))
    assert isinstance(data, xr.DataArray)
    for v in ("level", "latitude", "longitude", "step", "init_time_utc"):
        assert v in data.dims
    assert data.shape == (2, 10, 657, 1377, 3)
