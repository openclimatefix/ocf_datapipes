# OCF Datapipes
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

OCF's DataPipes for training and inference in Pytorch.

## Usage

These datapipes are designed to be composable and modular, and follow the same setup as for the in-built Pytorch
Datapipes. There are some great docs on how they can be composed and used [here](https://pytorch.org/data/main/examples.html).

End to end examples are given in `ocf_datapipes.training` and `ocf_datapipes.production`.


## Organization

This repo is organized as follows. The general flow of data loading and processing
goes from the `ocf_datapipes.load -> .select -> .transform.xarray -> .convert` and
then optionally `.transform.numpy`.

`training` and `production` contain datapipes that go through all the steps of
loading the config file, data, selecting and transforming data, and returning the
numpy data to the PyTorch dataloader.

Modules have their own README's as well to go into further detail.

```
.
└── ocf_datapipes/
    ├── batch/
    │   └── fake
    ├── config/
    │   └── convert/
    │       └── numpy/
    │           └── batch
    ├── experimental
    ├── fake
    ├── load/
    │   ├── gsp
    │   ├── nwp
    │   └── pv
    ├── production
    ├── select
    ├── training
    │   ├── datamodules
    ├── transform/
    │   ├── numpy/
    │   │   └── batch
    │   └── xarray/
    │       └── pv
    ├── utils/
    │   └── split
    └── validation
```

## Adding a new DataPipe
A general outline for a new DataPipe should go something
like this:

```python
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe

@functional_datapipe("<pipelet_name>")
class <PipeletName>IterDataPipe(IterDataPipe):
    def __init__(self):
        pass

    def __iter__(self):
        pass
```

### Below is a little more detailed example on how to create and join multiple datapipes.

```python

## The below code snippets have been picked from ocf_datapipes\training\pv_satellite_nwp.py file


# 1. read the configuration model for the dataset, detailing what kind of data is the dataset holding, e.g., pv, pv+satellite, pv+satellite+nwp, etc

    config_datapipe = OpenConfiguration(configuration)

# 2. create respective data pipes for pv, nwp and satellite

    pv_datapipe, pv_location_datapipe = (OpenPVFromNetCDF(pv=configuration.input_data.pv).pv_fill_night_nans().fork(2))

    nwp_datapipe = OpenNWP(configuration.input_data.nwp.nwp_zarr_path)

    satellite_datapipe = OpenSatellite(zarr_path=configuration.input_data.satellite.satellite_zarr_path)

# 3. pick all or random location data based on pv data pipeline

    location_datapipes = pv_location_datapipe.location_picker().fork(4, buffer_size=BUFFER_SIZE)

# 4. for the above picked locations get their respective spatial space slices from all the data pipes

    pv_datapipe, pv_time_periods_datapipe, pv_t0_datapipe = pv_datapipe.select_spatial_slice_meters(...)

    nwp_datapipe, nwp_time_periods_datapipe = nwp_datapipe.select_spatial_slice_pixels(...)

    satellite_datapipe, satellite_time_periods_datapipe = satellite_datapipe.select_spatial_slice_pixels(...)

# 5. get contiguous time period data for the above picked locations

    pv_time_periods_datapipe = pv_time_periods_datapipe.get_contiguous_time_periods(...)

    nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(...)

    satellite_time_periods_datapipe = satellite_time_periods_datapipe.get_contiguous_time_periods(...)

# 6. since all the datapipes have different sampling period for their data, lets find the time that is common between all the data pipes

    overlapping_datapipe = pv_time_periods_datapipe.select_overlapping_time_slice(secondary_datapipes=[nwp_time_periods_datapipe, satellite_time_periods_datapipe])

# 7. take time slices for the above overlapping time from all the data pipes

    pv_datapipe = pv_datapipe.select_time_slice(...)

    nwp_datapipe = nwp_datapipe.convert_to_nwp_target_time(...)

    satellite_datapipe = satellite_datapipe.select_time_slice(...)

# 8. Finally join all the data pipes together

    combined_datapipe = MergeNumpyModalities([nwp_datapipe, pv_datapipe, satellite_datapipe])


```

### Experimental DataPipes

For new datapipes being developed for new models or input modalities, to somewhat separate the more experimental and in
development datapipes from the ones better tested for production purposes, there is an `ocf_datapipes.experimental` namespace for
developing these more research-y datapipes. These datapipes might not, and probably are not, tested.
Once the model(s) using them are in production, they should be upgraded to one of the other namespaces and have tests added.

## Citation

If you find this code useful, please cite the following:

```
@misc{ocf_datapipes,
  author = {Bieker, Jacob, and Dudfield, Peter, and Kelly, Jack},
  title = {OCF Datapipes},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openclimatefix/ocf_datapipes}},
}
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt="Jacob Bieker"/><br /><sub><b>Jacob Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf_datapipes/commits?author=jacobbieker" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vrym2"><img src="https://avatars.githubusercontent.com/u/93340339?v=4?s=100" width="100px;" alt="Raj"/><br /><sub><b>Raj</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf_datapipes/commits?author=vrym2" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dfulu"><img src="https://avatars.githubusercontent.com/u/41546094?v=4?s=100" width="100px;" alt="James Fulton"/><br /><sub><b>James Fulton</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf_datapipes/commits?author=dfulu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rjmcoder"><img src="https://avatars.githubusercontent.com/u/19336259?v=4?s=100" width="100px;" alt="Ritesh Mehta"/><br /><sub><b>Ritesh Mehta</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf_datapipes/commits?author=rjmcoder" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
