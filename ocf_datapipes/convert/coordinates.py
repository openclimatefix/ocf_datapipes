from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.geospatial import (
    lat_lon_to_osgb,
    load_geostationary_area_definition_and_transform_latlon,
    osgb_to_lat_lon,
)


@functional_datapipe("convert_latlon_to_osgb")
class ConvertLatLonToOSGBIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for xr_data in self.source_datapipe:
            xr_data["x_osgb"], xr_data["y_osgb"] = lat_lon_to_osgb(
                latitude=xr_data["latitude"], longitude=xr_data["longitude"]
            )
            yield xr_data


@functional_datapipe("convert_osgb_to_latlon")
class ConvertOSGBToLatLonIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for xr_data in self.source_datapipe:
            xr_data["latitude"], xr_data["longitude"] = osgb_to_lat_lon(
                x=xr_data["x_osgb"], y=xr_data["y_osgb"]
            )
            yield xr_data


@functional_datapipe("convert_geostationary_to_latlon")
class ConvertGeostationaryToLatLonIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for xr_data in self.source_datapipe:
            transform = load_geostationary_area_definition_and_transform_latlon(xr_data)
            xr_data["latitude"], xr_data["longitude"] = transform(
                xx=xr_data["x_geostationary"], yy=xr_data["y_geostationary"]
            )
            yield xr_data
