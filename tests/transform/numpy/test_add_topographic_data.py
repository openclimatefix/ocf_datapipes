from ocf_datapipes.transform.numpy import AddTopographicData
from ocf_datapipes.transform.xarray import ReprojectTopography


def test_add_topo_data_hrvsatellite(sat_hrv_np_dp, topo_dp):
    topo_dp = ReprojectTopography(topo_dp)
    combined_dp = AddTopographicData(sat_hrv_np_dp, topo_datapipe=topo_dp)
    data = next(iter(combined_dp))
    assert data is not None
