from ocf_datapipes.transform.numpy_batch import AddTopographicData
from ocf_datapipes.transform.xarray import ReprojectTopography
import pytest

@pytest.mark.skip('Test not working')
def test_add_topo_data_hrvsatellite(sat_hrv_np_datapipe, topo_datapipe):
    # These datapipes are expected to yeild batches rather than samples for the following funcs
    topo_datapipe.batch(4).merge_numpy_batch()
    sat_hrv_np_datapipe = sat_hrv_np_datapipe.batch(4).merge_numpy_batch()

    topo_datapipe = ReprojectTopography(topo_datapipe)
    combined_datapipe = AddTopographicData(sat_hrv_np_datapipe, topo_datapipe=topo_datapipe)
    data = next(iter(combined_datapipe))
    assert data is not None
