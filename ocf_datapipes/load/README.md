# Data Loading

This module contains the code for loading the data off disk.
This data is always opened into Xarray `DataSet` or `DataArray` objects for further processing.
The opened data should only have dimensions renamed to the common format, and minimal processing done.
Any real processing should be performed in the `transform` module.