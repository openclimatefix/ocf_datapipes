"""Various conversion datapipes"""

from .coordinates import ConvertGeostationaryToLonLatIterDataPipe as ConvertGeostationaryToLonLat
from .coordinates import ConvertLonLatToOSGBIterDataPipe as ConvertLonLatToOSGB
from .coordinates import ConvertOSGBToLonLatIterDataPipe as ConvertOSGBToLonLat
from .numpy.gsp import ConvertGSPToNumpyIterDataPipe as ConvertGSPToNumpy
from .numpy.pv import ConvertPVToNumpyIterDataPipe as ConvertPVToNumpy
from .numpy_batch.gsp import (
    ConvertGSPToNumpyBatchIterDataPipe as ConvertGSPToNumpyBatch,
)
from .numpy_batch.nwp import (
    ConvertNWPToNumpyBatchIterDataPipe as ConvertNWPToNumpyBatch,
)
from .numpy_batch.pv import (
    ConvertPVToNumpyBatchIterDataPipe as ConvertPVToNumpyBatch,
)
from .numpy_batch.satellite import (
    ConvertSatelliteToNumpyBatchIterDataPipe as ConvertSatelliteToNumpyBatch,
)
from .numpy_batch.sensor import (
    ConvertSensorToNumpyBatchIterDataPipe as ConvertSensorToNumpyBatch,
)
from .numpy_batch.wind import (
    ConvertWindToNumpyBatchIterDataPipe as ConvertWindToNumpyBatch,
)
from .stack_xarray_to_numpy import StackXarrayIterDataPipe as StackXarray
