# Selecting Data

This module contains datapipes that deal with
the selection of data. They all take Xarray `DataSet` or `DataArray` as inputs
and return, usually, the same object with the selection applied.

## Exceptions to returning Xarray objects

- `location_picker.py` returns a `Location` object containing the x and y coordinates for the center of an example
- `select_overlapping_time_slice.py` returns a pandas DataFrame of time periods of the overlapping time slices
- `select_t0_time.py` returns a pandas Timestamp for the t0 time
