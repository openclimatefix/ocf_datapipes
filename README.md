# OCF Datapipes

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
