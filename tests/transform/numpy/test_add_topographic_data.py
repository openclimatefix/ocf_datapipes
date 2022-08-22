from ocf_datapipes.transform.numpy import AddTopographicData


def test_add_topo_data_hrvsatellite(sat_hrv_dp, topo_dp):
    combined_dp = AddTopographicData(sat_hrv_dp, topo_dp=topo_dp)
    data = next(iter(combined_dp))
    assert data is not None

