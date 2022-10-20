from ocf_datapipes.transform.xarray import PreProcessMetNet

def test_metnet_preprocess_satellite(sat_datapipe):
    PreProcessMetNet([sat_datapipe])