from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("downsample")
class DownsampleIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, y_coarsen: int, x_coarsen):
        self.source_dp = source_dp
        self.y_coarsen = y_coarsen
        self.x_coarsen = x_coarsen

    def __iter__(self):
        # TODO Change to lat/lon when doing that
        for xr_data in self.source_dp:
            yield xr_data.coarsen(
                y_osgb=self.y_coarsen,
                x_osgb=self.x_coarsen,
                boundary="trim",
            ).mean()
