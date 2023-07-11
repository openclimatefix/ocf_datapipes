"""Create the training/validation datapipe for training the PVNet Model"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenGSPFromDatabase
from ocf_datapipes.training.common import (
    create_t0_and_loc_datapipes,
    open_and_return_datapipes,
)
from ocf_datapipes.utils.consts import (
    NEW_NWP_MEAN,
    NEW_NWP_STD,
    RSS_MEAN,
    RSS_STD,
    BatchKey,
    NumpyBatch,
)

xr.set_options(keep_attrs=True)
logger = logging.getLogger("pvnet_datapipe")


def normalize_gsp(x):
    """Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.effective_capacity_mwp


def production_sat_scale(x):
    """Scale the production satellite data

    Args:
        x: Input DataArray

    Returns:
        Scaled DataArray
    """
    return x / 1024


def pvnet_concat_gsp(gsp_dataarrays: List[xr.DataArray]):
    """This function is used to combine the split history and future gsp dataarrays.

    These are split inside the `slice_datapipes_by_time()` function below.

    Splitting them inside that function allows us to apply dropout to the
    history GSP whilst leaving the future GSP without NaNs.

    We recombine the history and future with this function to allow us to use the
    `MergeNumpyModalities()` datapipe without redefining the BatchKeys.

    The `pvnet` model was also written to use a GSP array which has historical and future
    and to split it out. These maintains that assumption.
    """
    return xr.concat(gsp_dataarrays, dim="time_utc")


def gsp_drop_national(x: Union[xr.DataArray, xr.Dataset]):
    """Drop entries for national PV output

    Args:
        x: Data source of gsp data

    Returns:
        Filtered data source
    """
    return x.where(x.gsp_id != 0, drop=True)


def fill_nans_in_arrays(batch: NumpyBatch) -> NumpyBatch:
    """Fills all NaN values in each np.ndarray in the batch dictionary with zeros.

    Operation is performed in-place on the batch.
    """
    logger.info("Filling Nans with zeros")
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            np.nan_to_num(v, copy=False, nan=0.0)
    return batch


class AddZeroedSatelliteData:
    """A callable class used to add zeroed-out satellite data to batches of data.

    This is useful
    to speed up batch loading if pre-training the output part of the network without satellite
    inputs.
    """

    def __init__(self, configuration: Configuration, is_hrv: bool = False):
        """A callable class used to add zeroed-out satellite data to batches of data.

        Args:
            configuration: Configuration object
            is_hrv: If False, non-HRV data is added by called function, else HRV.
        """

        self.configuration = configuration
        self.is_hrv = is_hrv

    def __call__(self, batch: NumpyBatch) -> NumpyBatch:
        """Add zeroed-out satellite data to batch with shape accoriding to supplied configuration.

        Batch is modified in-place and returned.

        Args:
            batch: Numpy batch of input data.
        """

        variable = "hrvsatellite" if self.is_hrv else "satellite"

        satellite_config = getattr(self.configuration.input_data, variable)

        n_channels = len(getattr(satellite_config, f"{variable}_channels"))
        height = getattr(satellite_config, f"{variable}_image_size_pixels_height")
        width = getattr(satellite_config, f"{variable}_image_size_pixels_width")

        sequence_len = satellite_config.history_minutes // 5 + 1 - 3

        batch[getattr(BatchKey, f"{variable}_actual")] = np.zeros(
            (sequence_len, n_channels, height, width)
        )

        return batch


class AddZeroedNWPData:
    """A callable class used to add zeroed-out NWP data to batches of data.

    This is useful to speed up batch loading if pre-training the output part of the network without
    NWP inputs.
    """

    def __init__(self, configuration: Configuration):
        """A callable class used to add zeroed-out NWP data to batches of data.

        Args:
            configuration: Configuration object
        """
        self.configuration = configuration

    def __call__(self, batch: NumpyBatch) -> NumpyBatch:
        """Add zeroed-out NWP data to batch with shape accoriding to supplied configuration.

        Batch is modified in-place and returned.

        Args:
            batch: Numpy batch of input data.
        """

        config = self.configuration.input_data.nwp

        n_channels = len(config.nwp_channels)
        height = config.nwp_image_size_pixels_height
        width = config.nwp_image_size_pixels_width

        sequence_len = config.history_minutes // 60 + config.forecast_minutes // 60 + 1

        batch[BatchKey.nwp] = np.zeros((sequence_len, n_channels, height, width))

        return batch


class DatapipeKeyForker:
    """ "Internal helper function to track forking of a datapipe."""

    def __init__(self, keys: List, datapipe: IterDataPipe):
        """Internal helper function to track forking of a datapipe.

        As forks are returned, this object tracks the keys left and returns the final copy of the
        datapipe when the last key is requested. This makes multiple forking easier and ensures
        closure.

        Args:
            keys: List of keys for which datapipe duplication is required.
            datapipe: Datapipe which will be forked for each ket
        """
        self.keys_left = keys
        self.datapipe = datapipe

    def __call__(self, key):
        """ "Returns a fork of `self.datapipe` and tracks a the keys left to ensure closure.

        Args:
            key: key to remove from `self.keys_left`. If `key` is None then an extra copy is made
            without affecting `self.keys_left`.
        """
        if len(self.keys_left) == 0:
            raise ValueError(f"No keys left when requested key : {key}")
        if key is not None:
            self.keys_left.remove(key)
        if len(self.keys_left) > 0:
            self.datapipe, return_datapipe = self.datapipe.fork(2, buffer_size=5)
        else:
            return_datapipe = self.datapipe
        return return_datapipe

    def close(self):
        """Asserts that the keys have all been used."""
        assert len(self.keys_left) == 0


def _get_datapipes_dict(
    config_filename: str,
    block_sat: bool,
    block_nwp: bool,
    production: bool = False,
):
    # Load datasets
    datapipes_dict = open_and_return_datapipes(
        configuration_filename=config_filename,
        use_gsp=(not production),
        use_pv=False,
        use_sat=not block_sat,  # Only loaded if we aren't replacing them with zeros
        use_hrv=False,
        use_nwp=not block_nwp,  # Only loaded if we aren't replacing them with zeros
        use_topo=False,
    )
    if production:
        configuration: Configuration = datapipes_dict["config"]

        datapipes_dict["gsp"] = OpenGSPFromDatabase().add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )
        datapipes_dict["sat"] = datapipes_dict["sat"].map(production_sat_scale)

    return datapipes_dict


def construct_loctime_pipelines(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    block_sat: bool = False,
    block_nwp: bool = False,
) -> Tuple[IterDataPipe, IterDataPipe]:
    """Construct location and time pipelines for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time for time datapipe.
        end_time: Maximum time for time datapipe.
        block_sat: Whether to load zeroes for satellite data.
        block_nwp: Whether to load zeroes for NWP data.
    """

    datapipes_dict = _get_datapipes_dict(
        config_filename,
        block_sat,
        block_nwp,
    )

    # Pull out config file
    config = datapipes_dict.pop("config")

    # We sample time and space of other data using GSP time and space coordinates, so filter GSP
    # data first amd this is carried through
    datapipes_dict["gsp"] = datapipes_dict["gsp"].map(gsp_drop_national)
    if (start_time is not None) or (end_time is not None):
        datapipes_dict["gsp"] = datapipes_dict["gsp"].select_train_test_time(start_time, end_time)

    # Get overlapping time periods
    location_pipe, t0_datapipe = create_t0_and_loc_datapipes(
        datapipes_dict,
        configuration=config,
        key_for_t0="gsp",
        shuffle=True,
        nwp_max_t0_offset=minutes(180),
    )

    return location_pipe, t0_datapipe


def minutes(num_mins: int):
    """Timedelta of a number of minutes.

    Args:
        num_mins: Minutes timedelta.
    """
    return timedelta(minutes=num_mins)


def slice_datapipes_by_time(
    datapipes_dict: Dict,
    t0_datapipe: IterDataPipe,
    configuration: Configuration,
    production: bool = False,
) -> None:
    """
    Modifies a dictionary of datapipes in-place to yield samples for given times t0.

    The NWP data* will be at least 90 minutes stale (i.e. as if it takes 90 minutes for the foreast
    to become available).

    The satellite data* is shaped so that the most recent can be 15 minutes before t0. However, 50%
    of the time dropout is applied so that the most recent field is between 45 and 20 minutes before
    t0. When dropped out like this, the values after this selected dropout time are set to NaN.

    The HRV data* is similar to the satellite data and if both are included they drop out
    simulataneously.

    The GSP data is split into "gsp" and "gsp_future" keys. 10% of the time the gsp value for time
    t0, which occurs under the "gsp" key, is set to NaN

    The PV data* is also split it "pv" and "pv_future" keys.

    * if included

    n.b. PV and HRV are included in this function, but not yet in the rest of the pvnet pipeline.
    This is mostly for demonstratio purposes of how the dropout might be applied.

    Args:
        datapipes_dict: Dictionary of used datapipes and t0 ones
        t0_datapipe: Datapipe which yields t0 times for sample
        configuration: Configuration object.
        production: Whether constucting pipeline for production inference. No dropout is used if
            True.

    """

    conf_in = configuration.input_data

    # Use DatapipeKeyForker to avoid forking t0_datapipe too many times, or leaving any forks unused
    fork_keys = {k for k in datapipes_dict.keys() if k not in ["topo"]}
    get_t0_datapipe = DatapipeKeyForker(fork_keys, t0_datapipe)

    sat_and_hrv_dropout_kwargs = dict(
        # Satellite is either 30 minutes or 60 minutes delayed
        dropout_timedeltas=[minutes(-60), minutes(-30)],
        dropout_frac=0 if production else 1.0,
    )

    # Satellite data never more recent than t0-30mins
    if production:
        sat_delay = minutes(-configuration.input_data.satellite.live_delay_minutes)
    else:
        sat_delay = minutes(-30)

    if "nwp" in datapipes_dict:
        datapipes_dict["nwp"] = datapipes_dict["nwp"].convert_to_nwp_target_time_with_dropout(
            t0_datapipe=get_t0_datapipe("nwp"),
            sample_period_duration=minutes(60),
            history_duration=minutes(conf_in.nwp.history_minutes),
            forecast_duration=minutes(conf_in.nwp.forecast_minutes),
            # The NWP forecast will always be at least 180 minutes stale
            dropout_timedeltas=[minutes(-180)],
            dropout_frac=0 if production else 1.0,
        )

    if "sat" in datapipes_dict:
        # Take time slices of sat data
        datapipes_dict["sat"] = datapipes_dict["sat"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(5),
            interval_start=minutes(-conf_in.satellite.history_minutes),
            interval_end=sat_delay,
            fill_selection=production,
        )

        # Generate randomly sampled dropout times
        sat_dropout_time_datapipe = get_t0_datapipe("sat").select_dropout_time(
            **sat_and_hrv_dropout_kwargs
        )

        if "hrv" in datapipes_dict:
            # Make dropout-time copy for hrv if included in data.
            # HRV and non-HRV will dropout simultaneously.
            sat_dropout_time_datapipe, hrv_dropout_time_datapipe = sat_dropout_time_datapipe.fork(
                2, buffer_size=5
            )

        # Apply the dropout
        datapipes_dict["sat"] = datapipes_dict["sat"].apply_dropout_time(
            dropout_time_datapipe=sat_dropout_time_datapipe,
        )

    if "hrv" in datapipes_dict:
        if "sat" not in datapipes_dict:
            # Generate randomly sampled dropout times
            # This is shared with sat if sat included
            hrv_dropout_time_datapipe = get_t0_datapipe(None).select_dropout_time(
                **sat_and_hrv_dropout_kwargs
            )

        datapipes_dict["hrv"] = datapipes_dict["hrv"].select_time_slice(
            t0_datapipe=get_t0_datapipe("hrv"),
            sample_period_duration=minutes(5),
            interval_start=minutes(-conf_in.hrvsatellite.history_minutes),
            interval_end=sat_delay,
            fill_selection=production,
        )

        # Apply the dropout
        datapipes_dict["hrv"] = datapipes_dict["hrv"].apply_dropout_time(
            dropout_time_datapipe=hrv_dropout_time_datapipe,
        )

    if "pv" in datapipes_dict:
        datapipes_dict["pv"], dp = datapipes_dict["pv"].fork(2, buffer_size=5)

        datapipes_dict["pv_future"] = dp.select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(5),
            interval_start=minutes(5),
            interval_end=minutes(conf_in.pv.forecast_minutes),
            fill_selection=production,
        )

        datapipes_dict["pv"] = datapipes_dict["pv"].select_time_slice(
            t0_datapipe=get_t0_datapipe("pv"),
            sample_period_duration=minutes(5),
            interval_start=minutes(-conf_in.pv.history_minutes),
            interval_end=minutes(0),
            fill_selection=production,
        )

    if "gsp" in datapipes_dict:
        datapipes_dict["gsp"], dp = datapipes_dict["gsp"].fork(2, buffer_size=5)

        datapipes_dict["gsp_future"] = dp.select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(30),
            interval_start=minutes(30),
            interval_end=minutes(conf_in.gsp.forecast_minutes),
            fill_selection=production,
        )

        datapipes_dict["gsp"] = datapipes_dict["gsp"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(30),
            interval_start=-minutes(conf_in.gsp.history_minutes),
            interval_end=minutes(0),
            fill_selection=production,
        )

        # Dropout on the GSP, but not the future GSP
        gsp_dropout_time_datapipe = get_t0_datapipe("gsp").select_dropout_time(
            # GSP data for time t0 may be missing. Only have value for t0-30mins
            dropout_timedeltas=[minutes(-30)],
            dropout_frac=0 if production else 0.1,
        )

        datapipes_dict["gsp"] = datapipes_dict["gsp"].apply_dropout_time(
            dropout_time_datapipe=gsp_dropout_time_datapipe,
        )

    get_t0_datapipe.close()

    return


def construct_sliced_data_pipeline(
    config_filename: str,
    location_pipe: IterDataPipe,
    t0_datapipe: IterDataPipe,
    block_sat: bool = False,
    block_nwp: bool = False,
    production: bool = False,
    check_satellite_no_zeros: bool = False,
) -> IterDataPipe:
    """Constructs data pipeline for the input data config file.

    This yields samples from the location and time datapipes.

    Args:
        config_filename: Path to config file.
        location_pipe: Datapipe yielding locations.
        t0_datapipe: Datapipe yielding times.
        block_sat: Whether to load zeroes for satellite data.
        block_nwp: Whether to load zeroes for NWP data.
        production: Whether constucting pipeline for production inference.
        check_satellite_no_zeros: Whether to check that satellite data has no zeros.
    """

    assert not (production and (block_sat or block_nwp))

    datapipes_dict = _get_datapipes_dict(
        config_filename,
        block_sat,
        block_nwp,
        production=production,
    )

    configuration = datapipes_dict.pop("config")

    # Unpack for convenience
    conf_sat = configuration.input_data.satellite
    conf_nwp = configuration.input_data.nwp

    # Slice all of the datasets by time - this is an in-place operation
    slice_datapipes_by_time(datapipes_dict, t0_datapipe, configuration, production)

    # Spatially slice, normalize, and convert data to numpy arrays
    numpy_modalities = []

    if "nwp" in datapipes_dict:
        nwp_datapipe = datapipes_dict["nwp"]

        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
            location_pipe_copy,
            roi_height_pixels=conf_nwp.nwp_image_size_pixels_height,
            roi_width_pixels=conf_nwp.nwp_image_size_pixels_width,
            x_dim_name="x_osgb",
            y_dim_name="y_osgb",
        )
        nwp_datapipe = nwp_datapipe.normalize(mean=NEW_NWP_MEAN, std=NEW_NWP_STD)
        numpy_modalities.append(nwp_datapipe.convert_nwp_to_numpy_batch())

    if "sat" in datapipes_dict:
        sat_datapipe = datapipes_dict["sat"]

        location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
        sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
            location_pipe_copy,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
        )
        sat_datapipe = sat_datapipe.normalize(mean=RSS_MEAN, std=RSS_STD)
        numpy_modalities.append(sat_datapipe.convert_satellite_to_numpy_batch())

    # GSP always assumed to be in data
    location_pipe, location_pipe_copy = location_pipe.fork(2, buffer_size=5)
    gsp_future_datapipe = datapipes_dict["gsp_future"]
    gsp_future_datapipe = gsp_future_datapipe.select_spatial_slice_meters(
        location_datapipe=location_pipe_copy,
        roi_height_meters=1,
        roi_width_meters=1,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
        dim_name="gsp_id",
    )

    gsp_datapipe = datapipes_dict["gsp"]
    gsp_datapipe = gsp_datapipe.select_spatial_slice_meters(
        location_datapipe=location_pipe,
        roi_height_meters=1,
        roi_width_meters=1,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
        dim_name="gsp_id",
    )

    # Recombine GSP arrays - see function doc for further explanation
    gsp_datapipe = gsp_datapipe.zip_ocf(gsp_future_datapipe).map(pvnet_concat_gsp)
    gsp_datapipe = gsp_datapipe.normalize(normalize_fn=normalize_gsp)

    numpy_modalities.append(gsp_datapipe.convert_gsp_to_numpy_batch())

    logger.debug("Combine all the data sources")
    combined_datapipe = MergeNumpyModalities(numpy_modalities).add_sun_position(modality_name="gsp")

    if block_sat and conf_sat != "":
        sat_block_func = AddZeroedSatelliteData(configuration)
        combined_datapipe = combined_datapipe.map(sat_block_func)

    if block_nwp and conf_nwp != "":
        nwp_block_func = AddZeroedNWPData(configuration)
        combined_datapipe = combined_datapipe.map(nwp_block_func)

    logger.info("Filtering out samples with no data")
    if check_satellite_no_zeros:
        # in production we don't want any nans in the satellite data
        combined_datapipe = combined_datapipe.map(check_nans_in_satellite_data)

    combined_datapipe = combined_datapipe.map(fill_nans_in_arrays)

    return combined_datapipe


def pvnet_datapipe(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    block_sat: bool = False,
    block_nwp: bool = False,
) -> IterDataPipe:
    """
    Construct pvnet pipeline for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time at which a sample can be selected.
        end_time: Maximum time at which a sample can be selected.
        block_sat: Whether to load zeroes for satellite data.
        block_nwp: Whether to load zeroes for NWP data.
    """
    logger.info("Constructing pvnet pipeline")

    # Open datasets from the config and filter to useable location-time pairs
    location_pipe, t0_datapipe = construct_loctime_pipelines(
        config_filename,
        start_time,
        end_time,
        block_sat,
        block_nwp,
    )

    # Shard after we have the loc-times. These are already shuffled so no need to shuffle again
    location_pipe = location_pipe.sharding_filter()
    t0_datapipe = t0_datapipe.sharding_filter()

    # In this function we re-open the datasets to make a clean separation before/after sharding
    # This function
    datapipe = construct_sliced_data_pipeline(
        config_filename,
        location_pipe,
        t0_datapipe,
        block_sat,
        block_nwp,
    )

    return datapipe


def check_nans_in_satellite_data(batch: NumpyBatch) -> NumpyBatch:
    """
    Check if there are any Nans values in the satellite data.
    """
    if np.any(np.isnan(batch[BatchKey.satellite_actual])):
        logger.error("Found nans values in satellite data")

        for t in range(batch[BatchKey.satellite_actual].shape[1]):
            if np.any(np.isnan(batch[BatchKey.satellite_actual][:, t])):
                logger.error(f"Found nans values in satellite data at time index {t}")

        raise ValueError("Found nans values in satellite data")

    return batch
