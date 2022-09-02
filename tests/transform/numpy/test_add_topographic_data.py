from ocf_datapipes.transform.numpy import AddTopographicData
from ocf_datapipes.transform.xarray import ReprojectTopography


def test_add_topo_data_hrvsatellite(sat_hrv_np_datapipe, topo_datapipe):
    topo_datapipe = ReprojectTopography(topo_datapipe)
    combined_datapipe = AddTopographicData(sat_hrv_np_datapipe, topo_datapipe=topo_datapipe)
    data = next(iter(combined_datapipe))
    assert data is not None
