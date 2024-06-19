"""Encodes the Fourier features for space and time"""

import logging
import warnings
from typing import Union

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey, NumpyBatch, NWPBatchKey, NWPNumpyBatch

logger = logging.getLogger(__name__)


@functional_datapipe("add_fourier_space_time")
class AddFourierSpaceTimeIterDataPipe(IterDataPipe):
    """Encodes the Fourier features for space and time"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        n_fourier_features_per_dim: int = 8,
    ):
        """
        Encodes the space and time fourier features

        Args:
            source_datapipe: Datapipe of NumpyBatch
            n_fourier_features_per_dim: Number of Fourier features per dimension
        """
        self.source_datapipe = source_datapipe
        self.n_fourier_features_per_dim = n_fourier_features_per_dim

    def __iter__(self):
        for np_batch in self.source_datapipe:
            yield add_spatial_and_temporal_fourier_features(
                np_batch=np_batch,
                n_fourier_features_per_dim=self.n_fourier_features_per_dim,
            )


def add_spatial_and_temporal_fourier_features(
    np_batch: NumpyBatch,
    n_fourier_features_per_dim: int = 8,
) -> NumpyBatch:
    """Add fourier features for x_osgb, y_osgb and time_utc."""
    # NWP is nested so needs to be done separately
    nwp_in_batch = BatchKey.nwp in np_batch
    if nwp_in_batch:
        # Pop and process NWPBatch. Add back in later
        nwp_batch = np_batch.pop(BatchKey.nwp)
        for nwp_source in nwp_batch.keys():
            _add_spatial_and_temporal_fourier_features(
                nwp_batch[nwp_source], n_fourier_features_per_dim
            )

    # Process other coords
    _add_spatial_and_temporal_fourier_features(np_batch, n_fourier_features_per_dim)

    # Add back in NWP maybe
    if nwp_in_batch:
        np_batch[BatchKey.nwp] = nwp_batch

    return np_batch


def _add_spatial_and_temporal_fourier_features(
    batch: Union[NumpyBatch, NWPNumpyBatch],
    n_fourier_features_per_dim: int,
) -> Union[NumpyBatch, NWPNumpyBatch]:
    """Adds fourier encodings in place to batch dict"""
    for key in list(batch.keys()):
        if key.name.endswith(("x_osgb", "y_osgb", "time_utc")):
            if isinstance(key, BatchKey):
                fourier_key = BatchKey[f"{key.name}_fourier"]
            elif isinstance(key, NWPBatchKey):
                fourier_key = NWPBatchKey[f"{key.name}_fourier"]
            else:
                raise ValueError(f"Unregognized key: {key}")

            normalized_coords = normalize_coords(batch[key])

            batch[fourier_key] = compute_fourier_features(
                normalized_coords, n_fourier_features=n_fourier_features_per_dim
            )
    return


def compute_fourier_features(
    array: np.ndarray,
    n_fourier_features: int = 8,
    min_freq: float = 2,
    max_freq: float = 8,
) -> np.ndarray:
    """Compute Fourier features for a single dimension, across all examples in a batch.

    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        array: np.ndarray with values roughly in the range [0, 1].
            The values don't have to be *exactly* in the range [0, 1] because sine and cosine
            handle values below 0 and above 2*pi.
            For the time dimension, the shape will be (batch_size, n_timesteps).
            For spatial dimensions, the shape might be (batch_size, length) or
            (batch_size, height, width).
            Although this function can cope with any shape `array`, with any number of dimensions.
        n_fourier_features: Total number of requested Fourier features. Must be an even number
            because half a sine and half are cosine.
        min_freq: If min_freq=2 and array is in the range [0, 1] then the lowest freq "wave" will
            go from -1 to +1 across the dimension.
        max_freq: Maximum frequency for the fourier features

    Returns:
        fourier_features: An np.ndarray of the same dtype as `array`,
            with shape `array.shape + (n_fourier_features,)`. Fourier features with even indexes
            are sine. Odd indexes are cosine.
    """
    assert n_fourier_features % 2 == 0
    assert min_freq > 0
    assert max_freq > min_freq

    div_term = np.linspace(
        start=min_freq,
        stop=max_freq,
        num=n_fourier_features // 2,
        dtype=array.dtype,
    )
    fourier_features = np.full(
        shape=array.shape + (n_fourier_features,),
        fill_value=np.nan,
        dtype=array.dtype,
    )

    radians = array * np.pi / 2
    radians = np.expand_dims(radians, axis=-1)
    radians_x_div_term = radians * div_term
    fourier_features[..., 1::2] = np.cos(radians_x_div_term)
    fourier_features[..., 0::2] = np.sin(radians_x_div_term)

    # Sanity check:
    if np.isfinite(array).all():
        assert np.isfinite(fourier_features).all()
    return fourier_features


def normalize_coords(
    coords: np.ndarray,
) -> np.ndarray:
    """Rescale the coords for a single dimension, across all modalities.

    Args:
        coords: Array of coords
    """
    batch_size = coords.shape[0]

    # min and max taken over these dims
    reduce_dims = tuple(range(1, len(coords.shape)))

    with warnings.catch_warnings():
        # We expect to encounter all-NaN slices (when a whole PV example is missing,
        # for example.)
        warnings.filterwarnings("ignore", "All-NaN slice encountered")

        min_per_example = np.nanmin(
            coords.reshape((batch_size, -1)), axis=reduce_dims, keepdims=True
        )

        max_per_example = np.nanmax(
            coords.reshape((batch_size, -1)), axis=reduce_dims, keepdims=True
        )

    assert np.isfinite(min_per_example).all()
    assert np.isfinite(max_per_example).all()

    normalized_coords = (coords - min_per_example) / (max_per_example - min_per_example)
    return normalized_coords
