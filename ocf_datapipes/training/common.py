"""Common functionality for datapipes"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import (
    OpenAWOSFromNetCDF,
    OpenConfiguration,
    OpenGSP,
    OpenGSPFromDatabase,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenPVFromPVSitesDB,
    OpenSatellite,
    OpenTopography,
)
from ocf_datapipes.utils.consts import BatchKey, NumpyBatch

logger = logging.getLogger(__name__)


def open_and_return_datapipes(
    configuration_filename: str,
    use_gsp: bool = True,
    use_nwp: bool = True,
    use_pv: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_topo: bool = True,
    use_sensor: bool = True,
    production: bool = False,
) -> dict[str, IterDataPipe]:
    """
    Open data sources given a configuration and return the list of datapipes

    Args:
        configuration_filename: Path to file to open
        use_nwp: Whether to use NWP data or not
        use_topo: Whether to use topographic data
        use_pv: Whether to open PV data
        use_hrv: Whether to open HRV satellite data
        use_sat: Whether to open non-HRV satellite data
        use_gsp: Whether to use GSP data or not
        use_sensor: Whether to use sensor data or not
        production: bool if this is for production or not

    Returns:
        List of datapipes corresponding to the datapipes to open
    """
    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Check which modalities to use
    conf_in = configuration.input_data
    use_nwp = use_nwp and (conf_in.nwp is not None) and (conf_in.nwp.nwp_zarr_path != "")
    use_pv = (
        use_pv and (conf_in.pv is not None) and (conf_in.pv.pv_files_groups[0].pv_filename != "")
    )
    use_sat = (
        use_sat
        and (conf_in.satellite is not None)
        and (conf_in.satellite.satellite_zarr_path != "")
    )
    use_hrv = (
        use_hrv
        and (conf_in.hrvsatellite is not None)
        and (conf_in.hrvsatellite.hrvsatellite_zarr_path != "")
    )
    use_topo = (
        use_topo
        and (conf_in.topographic is not None)
        and (conf_in.topographic.topographic_filename != "")
    )
    use_gsp = use_gsp and (conf_in.gsp is not None) and (conf_in.gsp.gsp_zarr_path != "")
    use_sensor = (
        use_sensor and (conf_in.sensor is not None) and (conf_in.sensor.sensor_filename != "")
    )

    logger.debug(
        f"GSP: {use_gsp} NWP: {use_nwp} Sat: {use_sat},"
        f" HRV: {use_hrv} PV: {use_pv} Topo: {use_topo}"
        f" Sensor: {use_sensor}"
    )

    used_datapipes = {}

    used_datapipes["config"] = configuration

    # Load GSP national data
    if use_gsp:
        logger.debug("Opening GSP Data")
        gsp_datapipe = OpenGSP(
            gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )

        used_datapipes["gsp"] = gsp_datapipe

    # Load NWP data
    if use_nwp:
        logger.debug("Opening NWP Data")
        nwp_datapipe = (
            OpenNWP(
                configuration.input_data.nwp.nwp_zarr_path,
                provider=configuration.input_data.nwp.nwp_provider,
            )
            .select_channels(configuration.input_data.nwp.nwp_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(hours=1),
                history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            )
        )

        used_datapipes["nwp"] = nwp_datapipe

    if use_sat:
        logger.debug("Opening Satellite Data")
        sat_datapipe = (
            OpenSatellite(
                configuration.input_data.satellite.satellite_zarr_path,
                use_15_minute_data_if_needed=production,
            )
            .select_channels(configuration.input_data.satellite.satellite_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(
                    minutes=configuration.input_data.satellite.history_minutes
                ),
            )
        )

        used_datapipes["sat"] = sat_datapipe

    if use_hrv:
        logger.debug("Opening HRV Satellite Data")
        sat_hrv_datapipe = OpenSatellite(
            configuration.input_data.hrvsatellite.hrvsatellite_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
        )

        used_datapipes["hrv"] = sat_hrv_datapipe

    if use_pv:
        logger.debug("Opening PV")
        pv_datapipe = OpenPVFromNetCDF(
            pv=configuration.input_data.pv
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        )

        used_datapipes["pv"] = pv_datapipe

    if use_sensor:
        logger.debug("Opening Sensor Data")
        sensor_datapipe = OpenAWOSFromNetCDF(
            configuration.input_data.sensor
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.sensor.history_minutes),
        )

        used_datapipes["sensor"] = sensor_datapipe

    if use_topo:
        logger.debug("Opening Topographic Data")
        topo_datapipe = OpenTopography(configuration.input_data.topographic.topographic_filename)

        used_datapipes["topo"] = topo_datapipe

    return used_datapipes


def get_and_return_overlapping_time_periods_and_t0(used_datapipes: dict, key_for_t0: str = "gsp"):
    """
    Takes datapipes and obtains the overlapping time periods + t0 time datapipes

    Args:
        used_datapipes: Dictionary of datapipes to compute the time intersection of
        key_for_t0: Key to use for the t0 datapipe

    Returns:
        Dictionary of datapipes with the proper time slices selected
    """
    datapipes_for_time_periods = []  # Using later to compute intersections
    datapipes_to_return = {}  # Returned along with original ones
    t0_datapipe = None
    configuration = used_datapipes.pop("config")
    for key, datapipe in used_datapipes.items():
        if "topo" in key:
            continue
        if key_for_t0 in key:
            forked_datapipes = datapipe.fork(3, buffer_size=100)
            t0_datapipe = forked_datapipes[2]
        else:
            forked_datapipes = datapipe.fork(2, buffer_size=100)
        datapipes_to_return[key] = forked_datapipes[0]
        if "nwp" == key:
            time_periods_datapipe = forked_datapipes[1].get_contiguous_time_periods(
                sample_period_duration=timedelta(hours=3),  # Init times are 3 hours apart
                history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
                forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
                time_dim="init_time_utc",
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "sat" == key:
            time_periods_datapipe = forked_datapipes[1].get_contiguous_time_periods(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(
                    minutes=configuration.input_data.satellite.history_minutes
                ),
                forecast_duration=timedelta(minutes=0),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "hrv" == key:
            time_periods_datapipe = forked_datapipes[1].get_contiguous_time_periods(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(
                    minutes=configuration.input_data.hrvsatellite.history_minutes
                ),
                forecast_duration=timedelta(minutes=0),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "pv" == key:
            time_periods_datapipe = forked_datapipes[1].get_contiguous_time_periods(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
                forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)
        if "gsp" == key:
            time_periods_datapipe = forked_datapipes[1].get_contiguous_time_periods(
                sample_period_duration=timedelta(minutes=30),
                history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
                forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "sensor" == key:
            time_periods_datapipe = forked_datapipes[1].get_contiguous_time_periods(
                sample_period_duration=timedelta(minutes=30),
                history_duration=timedelta(minutes=configuration.input_data.sensor.history_minutes),
                forecast_duration=timedelta(
                    minutes=configuration.input_data.sensor.forecast_minutes
                ),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

    # Now have the forked ones
    # find joint overlapping timer periods
    logger.debug("Getting joint time periods")
    overlapping_datapipe = datapipes_for_time_periods[0].select_overlapping_time_slice(
        secondary_datapipes=datapipes_for_time_periods[1:],
    )

    # select time periods
    t0_datapipe = t0_datapipe.select_time_periods(time_periods=overlapping_datapipe)

    num_t0_datapipes = len(datapipes_to_return.keys())  # One for each input
    t0_datapipes = t0_datapipe.select_t0_time(return_all_times=False).fork(
        num_t0_datapipes, buffer_size=100
    )

    for i, key in enumerate(list(datapipes_to_return.keys())):
        datapipes_to_return[key + "_t0"] = t0_datapipes[i]

    # Re-add config for later
    datapipes_to_return["config"] = configuration
    if "topo" in used_datapipes.keys():
        datapipes_to_return["topo"] = used_datapipes["topo"]
    return datapipes_to_return


def normalize_gsp(x):
    """Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.effective_capacity_mwp


def normalize_pv(x):
    """Normalize the PV data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return (x / x.nominal_capacity_wp).clip(None, 5)


def production_sat_scale(x):
    """Scale the production satellite data

    Args:
        x: Input DataArray

    Returns:
        Scaled DataArray
    """
    return x / 1024


def concat_xr_time_utc(gsp_dataarrays: List[xr.DataArray]):
    """This function is used to combine the split history and future gsp/pv dataarrays.

    These are split inside the `slice_datapipes_by_time()` function below.

    Splitting them inside that function allows us to apply dropout to the
    history GSP/PV whilst leaving the future GSP/PV without NaNs.

    We recombine the history and future with this function to allow us to use the
    `MergeNumpyModalities()` datapipe without redefining the BatchKeys.

    The `pvnet` model was also written to use a GSP/PV array which has historical and future
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


@functional_datapipe("pvnet_select_pv_by_ml_id")
class PVNetSelectPVbyMLIDIterDataPipe(IterDataPipe):
    """Select specific set of PV systems by ML ID."""

    def __init__(self, source_datapipe: IterDataPipe, ml_ids: np.array):
        """Select specific set of PV systems by ML ID.

        Args:
            source_datapipe: Datapipe emitting PV xarray data
            ml_ids: List-like of ML IDs to select

        Returns:
            Filtered data source
        """
        self.source_datapipe = source_datapipe
        self.ml_ids = ml_ids

    def __iter__(self):
        for x in self.source_datapipe:
            # Check for missing IDs
            ml_ids_not_in_data = ~np.isin(self.ml_ids, x.ml_id)
            if ml_ids_not_in_data.any():
                missing_ml_ids = np.array(self.ml_ids)[ml_ids_not_in_data]
                logger.warning(
                    f"The following ML IDs were mising in the PV site-level input data: "
                    f"{missing_ml_ids}. The values for these IDs will be set to NaN."
                )

            x_filtered = (
                # Many ML-IDs are null, so filter first
                x.where(~x.ml_id.isnull(), drop=True)
                # Swap dimensions so we can select by ml_id coordinate
                .swap_dims({"pv_system_id": "ml_id"})
                # Select IDs - missing IDs are given NaN values
                .reindex(ml_id=self.ml_ids)
                # Swap back dimensions
                .swap_dims({"ml_id": "pv_system_id"})
            )
            yield x_filtered


def fill_nans_in_pv(x: Union[xr.DataArray, xr.Dataset]):
    """Fill NaNs in PV data with the value -1

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x.fillna(-1)


def fill_nans_in_arrays(batch: NumpyBatch) -> NumpyBatch:
    """Fills all NaN values in each np.ndarray in the batch dictionary with zeros.

    Operation is performed in-place on the batch.
    """
    logger.info("Filling Nans with zeros")
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            np.nan_to_num(v, copy=False, nan=0.0)
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
    production: bool = False,
):
    # Load datasets
    datapipes_dict = open_and_return_datapipes(
        configuration_filename=config_filename,
        use_gsp=(not production),
        use_pv=(not production),
        use_sat=True,
        use_hrv=True,
        use_nwp=True,
        use_topo=True,
        use_sensor=True,
        production=production,
    )

    config: Configuration = datapipes_dict["config"]

    if production:
        datapipes_dict["gsp"] = OpenGSPFromDatabase().add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=config.input_data.gsp.history_minutes),
        )
        if "sat" in datapipes_dict:
            datapipes_dict["sat"] = datapipes_dict["sat"].map(production_sat_scale)
        if "pv" in datapipes_dict:
            datapipes_dict["pv"] = OpenPVFromPVSitesDB(config.input_data.pv.history_minutes)

    if "pv" in datapipes_dict and config.input_data.pv.pv_ml_ids != []:
        datapipes_dict["pv"] = datapipes_dict["pv"].pvnet_select_pv_by_ml_id(
            config.input_data.pv.pv_ml_ids
        )

    return datapipes_dict


def construct_loctime_pipelines(
    config_filename: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Tuple[IterDataPipe, IterDataPipe]:
    """Construct location and time pipelines for the input data config file.

    Args:
        config_filename: Path to config file.
        start_time: Minimum time for time datapipe.
        end_time: Maximum time for time datapipe.
    """

    datapipes_dict = _get_datapipes_dict(
        config_filename,
    )

    # Pull out config file
    config = datapipes_dict.pop("config")

    # We sample time and space of other data using GSP time and space coordinates, so filter GSP
    # data first amd this is carried through
    preferred_order_of_keys = [
        "gsp",
        "pv",
        "sensor",
    ]
    for key in preferred_order_of_keys:
        if key in datapipes_dict.keys():
            core_key = key
            break
    if core_key == "gsp":
        datapipes_dict[core_key] = datapipes_dict[core_key].map(gsp_drop_national)
    if (start_time is not None) or (end_time is not None):
        datapipes_dict[core_key] = datapipes_dict[core_key].select_train_test_time(
            start_time, end_time
        )

    # Get overlapping time periods
    location_pipe, t0_datapipe = create_t0_and_loc_datapipes(
        datapipes_dict,
        configuration=config,
        key_for_t0=core_key,
        shuffle=True,
        nwp_max_dropout_minutes=180,
        # Sometimes the forecast is only 4/day so 6 hour intervals - then we add 3-hour dropout
        nwp_max_staleness_minutes=60 * 9,
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
    if "sat" in datapipes_dict or "hrv" in datapipes_dict:
        sat_and_hrv_dropout_kwargs = dict(
            # Satellite is either 30 minutes or 60 minutes delayed in production.
            # Match during training
            dropout_timedeltas=[minutes(-60), minutes(-30)],
            dropout_frac=0 if production else 1.0,
        )

        sat_delay = minutes(-configuration.input_data.satellite.live_delay_minutes)

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
            max_steps_gap=2,
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
            max_steps_gap=2,
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
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(5),
            interval_start=minutes(-conf_in.pv.history_minutes),
            interval_end=minutes(0),
            fill_selection=production,
        )

        # Dropout on the PV, but not the future PV
        pv_dropout_time_datapipe = get_t0_datapipe("pv").select_dropout_time(
            # All PV data could be delayed by up to 30 minutes
            # (this does not stem from production - just setting for now)
            dropout_timedeltas=[minutes(m) for m in range(-30, 0, 5)],
            dropout_frac=0.1 if production else 1,
        )

        datapipes_dict["pv"] = datapipes_dict["pv"].apply_dropout_time(
            dropout_time_datapipe=pv_dropout_time_datapipe,
        )

        # Apply extra PV dropout using different delays per system and dropping out
        # entire PV systems
        # independently
        if not production:
            datapipes_dict["pv"].apply_pv_dropout(
                system_dropout_fractions=np.linspace(0, 0.2, 100),
                system_dropout_timedeltas=[minutes(m) for m in [-15, -10, -5, 0]],
            )

    if "sensor" in datapipes_dict:
        datapipes_dict["sensor"], dp = datapipes_dict["sensor"].fork(2, buffer_size=5)

        datapipes_dict["sensor_future"] = dp.select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(30),
            interval_start=minutes(30),
            interval_end=minutes(conf_in.sensor.forecast_minutes),
            fill_selection=production,
        )

        datapipes_dict["sensor"] = datapipes_dict["sensor"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(30),
            interval_start=minutes(-conf_in.sensor.history_minutes),
            interval_end=minutes(0),
            fill_selection=production,
        )

        # Dropout on the sensor, but not the future sensor
        sensor_dropout_time_datapipe = get_t0_datapipe("sensor").select_dropout_time(
            # All sensor data could be delayed by up to 30 minutes
            # (this does not stem from production - just setting for now)
            dropout_timedeltas=[minutes(m) for m in range(-30, 0, 5)],
            dropout_frac=0.1 if production else 1,
        )

        datapipes_dict["sensor"] = datapipes_dict["sensor"].apply_dropout_time(
            dropout_time_datapipe=sensor_dropout_time_datapipe,
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


def check_nans_in_satellite_data(batch: NumpyBatch) -> NumpyBatch:
    """
    Check if there are any Nans values in the satellite data.
    """
    if np.any(np.isnan(batch[BatchKey.satellite_actual])):
        logger.error("Found nans values in satellite data")

        logger.error(batch[BatchKey.satellite_actual].shape)

        # loop over time and channels
        for dim in [0, 1]:
            for t in range(batch[BatchKey.satellite_actual].shape[dim]):
                if dim == 0:
                    sate_data_one_step = batch[BatchKey.satellite_actual][t]
                else:
                    sate_data_one_step = batch[BatchKey.satellite_actual][:, t]
                nans = np.isnan(sate_data_one_step)

                if np.any(nans):
                    percent_nans = np.sum(nans) / np.prod(sate_data_one_step.shape) * 100

                    logger.error(
                        f"Found nans values in satellite data at index {t} ({dim=}). "
                        f"{percent_nans}% of values are nans"
                    )
                else:
                    logger.error(f"Found no nans values in satellite data at index {t} {dim=}")

        raise ValueError("Found nans values in satellite data")

    return batch


def add_selected_time_slices_from_datapipes(used_datapipes: dict):
    """
    Takes datapipes and t0 datapipes and returns the sliced datapipes

    Args:
        used_datapipes: Dictionary of used datapipes and t0 ones

    Returns:
        Dictionary of datapipes after the time slices are selected
    """
    datapipes_to_return = {}  # Returned along with original ones
    configuration = used_datapipes.pop("config")
    for key, datapipe in used_datapipes.items():
        if "topo" in key:
            continue
        if "_t0" in key:
            continue
        if "nwp" == key:
            datapipes_to_return[key] = datapipe.convert_to_nwp_target_time(
                t0_datapipe=used_datapipes[key + "_t0"],
                sample_period_duration=timedelta(hours=1),
                history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
                forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
            )

        if "sat" == key:
            datapipes_to_return[key] = datapipe.select_time_slice(
                t0_datapipe=used_datapipes[key + "_t0"],
                history_duration=timedelta(
                    minutes=configuration.input_data.satellite.history_minutes
                ),
                forecast_duration=timedelta(minutes=0),
                sample_period_duration=timedelta(minutes=5),
            )

        if "hrv" == key:
            datapipes_to_return[key] = datapipe.select_time_slice(
                t0_datapipe=used_datapipes[key + "_t0"],
                history_duration=timedelta(
                    minutes=configuration.input_data.hrvsatellite.history_minutes
                ),
                forecast_duration=timedelta(minutes=0),
                sample_period_duration=timedelta(minutes=5),
            )

        if "pv" == key:
            pv_1, pv_2 = used_datapipes[key + "_t0"].fork(2)
            pv_dp1, pv_dp2 = datapipe.fork(2)
            datapipes_to_return[key] = pv_dp1.select_time_slice(
                t0_datapipe=pv_1,
                history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
                forecast_duration=timedelta(minutes=0),
                sample_period_duration=timedelta(minutes=5),
            )
            datapipes_to_return[key + "_future"] = pv_dp2.select_time_slice(
                t0_datapipe=pv_2,
                history_duration=timedelta(minutes=0),
                forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
                sample_period_duration=timedelta(minutes=5),
            )

        if "sensor" == key:
            sensor_1, sensor_2 = used_datapipes[key + "_t0"].fork(2)
            sensor_dp1, sensor_dp2 = datapipe.fork(2)
            datapipes_to_return[key] = sensor_dp1.select_time_slice(
                t0_datapipe=sensor_1,
                history_duration=timedelta(minutes=configuration.input_data.sensor.history_minutes),
                forecast_duration=timedelta(minutes=0),
                sample_period_duration=timedelta(minutes=30),
            )
            datapipes_to_return[key + "_future"] = sensor_dp2.select_time_slice(
                t0_datapipe=sensor_2,
                history_duration=timedelta(minutes=0),
                forecast_duration=timedelta(
                    minutes=configuration.input_data.sensor.forecast_minutes
                ),
                sample_period_duration=timedelta(minutes=30),
            )

        if "gsp" == key:
            gsp_1, gsp_2 = used_datapipes[key + "_t0"].fork(2)
            gsp_dp1, gsp_dp2 = datapipe.fork(2)
            datapipes_to_return[key] = gsp_dp1.select_time_slice(
                t0_datapipe=gsp_1,
                history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
                forecast_duration=timedelta(minutes=0),
                sample_period_duration=timedelta(minutes=30),
            )
            datapipes_to_return[key + "_future"] = gsp_dp2.select_time_slice(
                t0_datapipe=gsp_2,
                history_duration=timedelta(minutes=0),
                forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
                sample_period_duration=timedelta(minutes=30),
            )
    if "topo" in used_datapipes.keys():
        datapipes_to_return["topo"] = used_datapipes["topo"]
    datapipes_to_return["config"] = configuration
    return datapipes_to_return


def create_t0_and_loc_datapipes(
    datapipes_dict: dict,
    configuration: Configuration,
    key_for_t0: str = "gsp",
    shuffle: bool = True,
    nwp_max_dropout_minutes: int = 0,
    nwp_max_staleness_minutes: int = 180,
):
    """
    Takes source datapipes and returns datapipes of appropriate sample pairs of locations and times.

    The (location, t0) pairs are sampled without replacement.

    Args:
        datapipes_dict: Dictionary of datapipes of input sources for which we want to select
            appropriate location and times.
        configuration: Configuration object for inputs.
        key_for_t0: Key to use for the t0 datapipe. Must be "gsp" or "pv".
        shuffle: Whether to use the internal shuffle function when yielding location times. Else
            location times will be heavily ordered.
        nwp_max_dropout_minutes: If using dropout on NWP, sometimes we have to go back to previous
            NWP init time. In order to accomodate for this possibility in selecting times, set
            `nwp_max_dropout_minutes` as the max NWP dropout delay you plan to use.
        nwp_max_staleness_minutes: Sets a limit on how stale an NWP init time is allowed to be
            whilst still being used to costruct an example

    Returns:
        location datapipe, t0 datapipe

    """
    assert key_for_t0 in datapipes_dict
    assert key_for_t0 in [
        "gsp",
        "pv",
        "sensor",
    ]
    assert nwp_max_staleness_minutes >= nwp_max_dropout_minutes

    contiguous_time_datapipes = []  # Used to store contiguous time periods from each data source

    datapipes_dict[key_for_t0], key_datapipe = datapipes_dict[key_for_t0].fork(2, buffer_size=5)

    for key in datapipes_dict.keys():
        if key in ["topo"]:
            continue

        elif key == "nwp":
            datapipes_dict["nwp"], datapipe_copy = datapipes_dict["nwp"].fork(2, buffer_size=5)

            # NWP is a forecast product so gets its own contiguous function
            time_periods = datapipe_copy.get_contiguous_time_periods_nwp(
                history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
                max_staleness=timedelta(minutes=nwp_max_staleness_minutes),
                max_dropout=timedelta(minutes=nwp_max_dropout_minutes),
                time_dim="init_time_utc",
            )

            contiguous_time_datapipes.append(time_periods)

        else:
            if key == "sat":
                sample_frequency = 5
                history_duration = configuration.input_data.satellite.history_minutes
                forecast_duration = 0
                time_dim = "time_utc"

            elif key == "hrv":
                sample_frequency = 5
                history_duration = configuration.input_data.hrvsatellite.history_minutes
                forecast_duration = 0
                time_dim = "time_utc"

            elif key == "pv":
                sample_frequency = 5
                history_duration = configuration.input_data.pv.history_minutes
                forecast_duration = configuration.input_data.pv.forecast_minutes
                time_dim = "time_utc"

            elif key == "sensor":
                sample_frequency = 30
                history_duration = configuration.input_data.sensor.history_minutes
                forecast_duration = configuration.input_data.sensor.forecast_minutes
                time_dim = "time_utc"

            elif key == "gsp":
                sample_frequency = 30
                history_duration = configuration.input_data.gsp.history_minutes
                forecast_duration = configuration.input_data.gsp.forecast_minutes
                time_dim = "time_utc"

            else:
                raise ValueError(f"Unexpected key: {key}")

            datapipes_dict[key], datapipe_copy = datapipes_dict[key].fork(2, buffer_size=5)

            time_periods = datapipe_copy.get_contiguous_time_periods(
                sample_period_duration=timedelta(minutes=sample_frequency),
                history_duration=timedelta(minutes=history_duration),
                forecast_duration=timedelta(minutes=forecast_duration),
                time_dim=time_dim,
            )

            contiguous_time_datapipes.append(time_periods)

    # Find joint overlapping contiguous time periods
    if len(contiguous_time_datapipes) > 1:
        logger.debug("Getting joint time periods")
        overlapping_datapipe = contiguous_time_datapipes[0].select_overlapping_time_slice(
            secondary_datapipes=contiguous_time_datapipes[1:],
        )
    else:
        logger.debug("Skipping getting joint time periods")
        overlapping_datapipe = contiguous_time_datapipes[0]

    # Select time periods and set length
    key_datapipe = key_datapipe.select_time_periods(time_periods=overlapping_datapipe)

    t0_loc_datapipe = key_datapipe.select_loc_and_t0(return_all=True, shuffle=shuffle)

    location_pipe, t0_datapipe = t0_loc_datapipe.unzip(sequence_length=2)

    return location_pipe, t0_datapipe
