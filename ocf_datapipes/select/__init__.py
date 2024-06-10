"""Selection datapipes"""

from .apply_pv_dropout import ApplyPVDropoutIterDataPipe as ApplyPVDropout
from .apply_standard_dropout import ApplyDropoutTimeIterDataPipe as ApplyDropoutTime
from .apply_standard_dropout import DrawDropoutTimeIterDataPipe as DrawDropoutTime
from .filter_channels import FilterChannelsIterDataPipe as FilterChannels
from .filter_gsp_ids import FilterGSPIDsIterDataPipe as FilterGSPIDs
from .filter_pv_sys_generating_overnight import (
    FilterPvSysGeneratingOvernightIterDataPipe as FilterPvSysGeneratingOvernight,
)
from .filter_pv_sys_with_only_nan_in_a_day import (
    FilterPVSystemsWithOnlyNanInADayIterDataPipe as FilterPVSystemsWithOnlyNanInADay,
)
from .filter_pv_systems_by_capacity import (
    FilterPVSystemsOnCapacityIterDataPipe as FilterPVSystemsOnCapacity,
)
from .filter_time_periods import FilterTimePeriodsIterDataPipe as FilterTimePeriods
from .filter_times import FilterTimesIterDataPipe as SelectTrainTestTimePeriod
from .filter_to_overlapping_time_periods import (
    FilterToOverlappingTimePeriodsIterDataPipe as FilterToOverlappingTimePeriods,
)
from .find_contiguous_t0_time_periods import (
    FindContiguousT0TimePeriodsIterDataPipe as FindContiguousT0TimePeriods,
)
from .find_contiguous_t0_time_periods import (
    FindContiguousT0TimePeriodsNWPIterDataPipe as FindContiguousT0TimePeriodsNWP,
)
from .pick_locations import PickLocationsIterDataPipe as PickLocations
from .pick_locations_and_t0_times import PickLocationsAndT0sIterDataPipe as PickLocationsAndT0s
from .pick_t0_times import PickT0TimesIterDataPipe as PickT0Times
from .select_id import SelectIDIterDataPipe as SelectID
from .select_non_nan_timestamps import SelectNonNaNTimesIterDataPipe as SelectNonNaNTimes
from .select_spatial_slice import SelectSpatialSliceMetersIterDataPipe as SelectSpatialSliceMeters
from .select_spatial_slice import SelectSpatialSlicePixelsIterDataPipe as SelectSpatialSlicePixels
from .select_time_slice import SelectTimeSliceIterDataPipe as SelectTimeSlice
from .select_time_slice_nwp import (
    SelectTimeSliceNWPIterDataPipe as SelectTimeSliceNWP,
)
