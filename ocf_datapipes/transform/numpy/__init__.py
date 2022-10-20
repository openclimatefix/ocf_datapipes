"""Numpy transforms"""
from .batch.add_length import AddLengthIterDataPipe as AddLength
from .batch.add_topographic_data import AddTopographicDataIterDataPipe as AddTopographicData
from .batch.add_zeroed_future_pv import AddZeroedFutureDataIterDataPipe as AddZeroedFutureData
from .batch.align_gsp_to_5_min import AlignGSPto5MinIterDataPipe as AlignGSPto5Min
from .batch.change_np_float32 import ChangeFloat32IterDataPipe as ChangeFloat32
from .batch.encode_space_time import EncodeSpaceTimeIterDataPipe as EncodeSpaceTime
from .batch.extend_timestamps_to_future import (
    ExtendTimestepsToFutureIterDataPipe as ExtendTimestepsToFuture,
)
from .batch.save_t0_time import SaveT0TimeIterDataPipe as SaveT0Time
from .batch.sun_position import AddSunPositionIterDataPipe as AddSunPosition
