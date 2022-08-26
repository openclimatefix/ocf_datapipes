import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("normalize")
class NormalizeIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_dp: IterDataPipe,
        mean=None,
        std=None,
        max_value=None,
        calculate_mean_std_from_example: bool = False,
        normalize_fn=lambda x: x / x.capacity_wp,
    ):
        self.source_dp = source_dp
        self.mean = mean
        self.std = std
        self.max_value = max_value
        self.calculate_mean_std_from_example = calculate_mean_std_from_example
        self.normalize_fn = normalize_fn

    def __iter__(self):
        for xr_data in self.source_dp:
            if self.mean is not None and self.std is not None:
                xr_data = xr_data - self.mean
                xr_data = xr_data / self.std
            elif self.max_value is not None:
                xr_data = xr_data / self.max_value
            elif self.calculate_mean_std_from_example:
                # For Topo data for example
                xr_data -= xr_data.mean().item()
                xr_data /= xr_data.std().item()
            else:
                xr_data = self.normalize_fn(xr_data)
            yield xr_data
