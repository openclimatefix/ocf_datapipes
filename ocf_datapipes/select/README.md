# Selecting Data

This module contains datapipes that deal with the selection of data. They all take Xarray `DataSet` or `DataArray` as inputs and return, usually, the same object with the selection applied.

## Exceptions to returning Xarray objects

- `filter_overlapping_time_slice.py` returns a pandas DataFrame of time periods of the overlapping time slices
- Files with preffix `pick_`

The following naming convention applies:
- `select_` is reserved for sample level selection, like selecting a window in time or space for a sample
- `filter_` is reserved for dataset level selection, like slicing out the train/test period, selecting channels, or systems by ID
- `pick_` is reserved for functions which take a dataset and yield locations and/or t0 times
