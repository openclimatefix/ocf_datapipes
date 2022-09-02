from ocf_datapipes.load import OpenTopography


def test_open_topo():
    topo_datapipe = OpenTopography(topo_filename="tests/data/europe_dem_2km_osgb.tif")
    metadata = next(iter(topo_datapipe))
    assert metadata is not None
