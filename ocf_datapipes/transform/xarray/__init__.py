"""Xarray transforms"""
from ocf_datapipes.transform.xarray.pv.ensure_n_pv_systems_per_example import (
    EnsureNPVSystemsPerExampleIterDataPipe as EnsureNPVSystemsPerExample,
)
from ocf_datapipes.transform.xarray.pv.pv_fill_nighttime_nans_with_zeros import (
    PVFillNightNansIterDataPipe as PVFillNightNans,
)
from ocf_datapipes.transform.xarray.pv.pv_infill_interpolate import (
    PVInterpolateInfillIterDataPipe as PVInterpolateInfill,
)
from ocf_datapipes.transform.xarray.pv.pv_power_rolling_window import (
    PVPowerRollingWindowIterDataPipe as PVPowerRollingWindow,
)
from ocf_datapipes.transform.xarray.pv.remove_pv_zero_examples import (
    PVPowerRemoveZeroDataIterDataPipe as PVPowerRemoveZeroData,
)

from .add_t0idx_and_sample_period_duration import (
    AddT0IdxAndSamplePeriodDurationIterDataPipe as AddT0IdxAndSamplePeriodDuration,
)
from .convert_pressure_levels_to_separate_variables import (
    ConvertPressureLevelsToSeparateVariablesIterDataPipe as ConvertPressureLevelsToSeparateVariables,  # noqa: E501
)
from .convert_satellite_to_int import (
    ConvertSatelliteToInt8IterDataPipe as ConvertSatelliteToInt8,
)
from .convert_to_numpy_stack import StackXarrayIterDataPipe as StackXarray
from .convert_to_nwp_target_times import (
    ConvertToNWPTargetTimeIterDataPipe as ConvertToNWPTargetTime,
)
from .create_sun_image import CreateSunImageIterDataPipe as CreateSunImage
from .create_time_image import CreateTimeImageIterDataPipe as CreateTimeImage
from .downsample import DownsampleIterDataPipe as Downsample
from .get_contiguous_time_periods import (
    GetContiguousT0TimePeriodsIterDataPipe as GetContiguousT0TimePeriods,
)
from .get_contiguous_time_periods import (
    GetContiguousT0TimePeriodsNWPIterDataPipe as GetContiguousT0TimePeriodsNWP,
)
from .gsp.create_gsp_image import CreateGSPImageIterDataPipe as CreateGSPImage
from .gsp.ensure_n_gsp_per_example import (
    EnsureNGSPSPerExampleIterDataPipe as EnsureNGSPSPerExampleIter,
)
from .gsp.remove_northern_gsp import RemoveNorthernGSPIterDataPipe as RemoveNorthernGSP
from .metnet_preprocessor import PreProcessMetNetIterDataPipe as PreProcessMetNet
from .normalize import NormalizeIterDataPipe as Normalize
from .nwp_dropout import (
    ConvertToNWPTargetTimeWithDropoutIterDataPipe as ConvertToNWPTargetTimeWithDropout,
)
from .pv.assign_daynight_status import (
    AssignDayNightStatusIterDataPipe as AssignDayNightStatus,
)
from .pv.create_pv_history_image import (
    CreatePVHistoryImageIterDataPipe as CreatePVHistoryImage,
)
from .pv.create_pv_image import CreatePVImageIterDataPipe as CreatePVImage
from .pv.create_pv_meta_image import (
    CreatePVMetadataImageIterDataPipe as CreatePVMetadataImage,
)
from .pv_dropout import ApplyPVDropoutIterDataPipe as ApplyPVDropout
from .remove_nans import RemoveNansIterDataPipe as RemoveNans
from .reproject_topographic_data import (
    ReprojectTopographyIterDataPipe as ReprojectTopography,
)
from .standard_dropout import ApplyDropoutTimeIterDataPipe as ApplyDropoutTime
from .standard_dropout import SelectDropoutTimeIterDataPipe as SelectDropoutTime
