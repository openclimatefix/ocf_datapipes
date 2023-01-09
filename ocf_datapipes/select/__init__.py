"""Selection datapipes"""
from .drop_national_gsp import DropGSPIterDataPipe as DropGSP
from .drop_pv_sys_generating_overnight import (
    DropPvSysGeneratingOvernightIterDataPipe as DropPvSysGeneratingOvernight,
)
from .drop_pv_sys_with_only_nan_in_a_day import (
    DropPVSystemsWithOnlyNanInADayIterDataPipe as DropPVSystemsWithOnlyNanInADay,
)
from .location_picker import LocationPickerIterDataPipe as LocationPicker
from .number_of_location import NumberOfLocationsrIterDataPipe as NumberOfLocations
from .offset_t0 import OffsetT0IterDataPipe as OffsetT0
from .remove_bad_pv_systems import RemoveBadPVSystemsIterDataPipe as RemoveBadSystems
from .select_channels import SelectChannelsIterDataPipe as SelectChannels
from .select_id import SelectIDIterDataPipe as SelectID
from .select_live_t0_time import SelectLiveT0TimeIterDataPipe as SelectLiveT0Time
from .select_live_time_slice import SelectLiveTimeSliceIterDataPipe as SelectLiveTimeSlice
from .select_overlapping_time_slices import (
    SelectOverlappingTimeSliceIterDataPipe as SelectOverlappingTimeSlice,
)
from .select_spatial_slice import SelectSpatialSliceMetersIterDataPipe as SelectSpatialSliceMeters
from .select_spatial_slice import SelectSpatialSlicePixelsIterDataPipe as SelectSpatialSlicePixels
from .select_t0_time import SelectT0TimeIterDataPipe as SelectT0Time
from .select_time_periods import SelectTimePeriodsIterDataPipe as SelectTimePeriods
from .select_time_slice import SelectTimeSliceIterDataPipe as SelectTimeSlice
from .select_train_test import SelectTrainTestTimePeriodsIterDataPipe as SelectTrainTestTimePeriod
from .trim_dates_with_insufficent_data import (
    TrimDatesWithInsufficentDataIterDataPipe as TrimDatesWithInsufficentData,
)
