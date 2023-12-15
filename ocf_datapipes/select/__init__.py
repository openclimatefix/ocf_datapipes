"""Selection datapipes"""
from .apply_pv_dropout import ApplyPVDropoutIterDataPipe as ApplyPVDropout
from .apply_standard_dropout import (
    SelectDropoutTimeIterDataPipe as SelectDropoutTime, 
    ApplyDropoutTimeIterDataPipe as ApplyDropoutTime
)

from .filter_channels import SelectChannelsIterDataPipe as SelectChannels
from .filter_gsp_ids import SelectGSPIDsIterDataPipe as SelectGSPIDs
from .filter_overlapping_time_slices import (
    SelectOverlappingTimeSliceIterDataPipe as SelectOverlappingTimeSlice,
)
from .filter_pv_sys_generating_overnight import (
    DropPvSysGeneratingOvernightIterDataPipe as DropPvSysGeneratingOvernight,
)
from .filter_pv_sys_with_only_nan_in_a_day import (
    DropPVSystemsWithOnlyNanInADayIterDataPipe as DropPVSystemsWithOnlyNanInADay,
)
from .filter_pv_systems_by_capacity import (
    SelectPVSystemsOnCapacityIterDataPipe as SelectPVSystemsOnCapacity
)
from .filter_times import SelectTrainTestTimePeriodsIterDataPipe as SelectTrainTestTimePeriod
from .filter_time_periods import SelectTimePeriodsIterDataPipe as SelectTimePeriods
from .filter_to_contiguous_time_periods import (
 GetContiguousT0TimePeriodsIterDataPipe as GetContiguousT0TimePeriods
)
from .filter_to_contiguous_time_periods import (
 GetContiguousT0TimePeriodsNWPIterDataPipe as GetContiguousT0TimePeriodsNWP
)

from .number_of_location import NumberOfLocationsrIterDataPipe as NumberOfLocations

from .pick_locations_and_t0_times import LocationT0PickerIterDataPipe as LocationT0Picker
from .pick_locations import LocationPickerIterDataPipe as LocationPicker
from .pick_t0_times import SelectT0TimeIterDataPipe as SelectT0Time

from .select_id import SelectIDIterDataPipe as SelectID
from .select_non_nan_timestamps import RemoveNansIterDataPipe as RemoveNans
from .select_spatial_slice import SelectSpatialSliceMetersIterDataPipe as SelectSpatialSliceMeters
from .select_spatial_slice import SelectSpatialSlicePixelsIterDataPipe as SelectSpatialSlicePixels
from .select_time_slice_nwp import (
    ConvertToNWPTargetTimeWithDropoutIterDataPipe as ConvertToNWPTargetTimeWithDropout
)
from .select_time_slice import SelectTimeSliceIterDataPipe as SelectTimeSlice

