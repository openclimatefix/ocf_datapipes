# Data Loading

This module contains the code for loading the data off disk.
This data is always opened into Xarray `DataSet` or `DataArray` objects for further processing.
The opened data should only have dimensions renamed to the common format, and minimal processing done.
Any real processing should be performed in the `transform` module.

## NWP

The NWP data is loaded into an ```IterDataPipe``` in `ocf_datapipes/load/nwp/nwp.py` using a provider from `ocf_datapipes/load/nwp/providers`. Providers open the data file and transform the data into a standardised format that datapipes use; normally this means having the following 5 dimensions: ```init_time_utc, step, channel, latitude, longitude```.

Example of loaded ECMWF data:

```
<xarray.DataArray 'ECMWF_BLAH' (init_time_utc: 1, step: 2, channel: 2,
                                latitude: 221, longitude: 221)> Size: 781kB
dask.array<transpose, shape=(1, 2, 2, 221, 221), dtype=float32, chunksize=(1, 2, 2, 221, 221), chunktype=numpy.ndarray>
Coordinates:
  * init_time_utc  (init_time_utc) datetime64[ns] 8B 2023-09-25T12:00:00
  * latitude       (latitude) float64 2kB 31.0 30.95 30.9 ... 20.1 20.05 20.0
  * longitude      (longitude) float64 2kB 68.0 68.05 68.1 ... 78.9 78.95 79.0
  * step           (step) timedelta64[ns] 16B 00:00:00 01:00:00
  * channel        (channel) <U5 40B 'dlwrf' 'dswrf'
Attributes:
    Conventions:             CF-1.7
    GRIB_centre:             ecmf
    GRIB_centreDescription:  European Centre for Medium-Range Weather Forecasts
    GRIB_subCentre:          0
    institution:             European Centre for Medium-Range Weather Forecasts
```

There are exceptions, e.g. ICON Global uses an isohedral grid, so it is differently organised and does not have ```latitude``` and ```longitude``` dimensions.

### Adding an NWP provider

1. Add a [provider].py file to `ocf_datapipes/load/nwp/providers` that uses `open_zarr_paths` from `ocf_datapipes.load.nwp.providers.utils` to load the file(s) and returns your data in the right shape, where the dimensions contain:
   - `init_time_utc`: when the data was initialised
   - `step`: distance from datapoint to its init_time
   - `channel`: list of variables
   - `latitude`: latitude
   - `longitude`: longitude

Add sanity checks to ensure time is unique and monotonic

2. Add your provider as an option to the `IterDataPipe` in `ocf_datapipes/load/nwp/nwp.py`

3. Add test data to `ocf_datapipes/tests/data` and create a test in `ocf_datapipes/tests/load/nwp/test_load_nwp.py`
Current tests include:
    - checking the loaded data is not None
    - checking all expected dimensions are present
