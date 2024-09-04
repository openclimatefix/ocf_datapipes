"""Common functionality for datapipes"""

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.config.model import Configuration, InputData
from ocf_datapipes.load import (
    OpenConfiguration,
    OpenGSP,
    OpenGSPFromDatabase,
    OpenMeteomaticsFromZarr,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenPVFromPVSitesDB,
    OpenSatellite,
    OpenWindFromNetCDF,
)
from ocf_datapipes.utils.utils import flatten_nwp_source_dict

try:
    from ocf_datapipes.load import OpenTopography

    # Rioxarray is sometimes a pain to install, so only load this if its installed
except ImportError:
    print(
        "Could not import OpenTopography," " this is probably becasye Rioxarray is not installed."
    )
    pass


logger = logging.getLogger(__name__)


def is_config_and_path_valid(
    use_flag: bool,
    config,
    filepath_resolver: Union[str, Callable[[InputData], str]],
) -> bool:
    """
    Checks if the given configuration should be used based on specific criteria.

    Args:
        use_flag (bool): Indicates whether to consider using the configuration.
        config (object): The configuration object to check.
        filepath_resolver (str or callable): Specifies how to access the file path within config;
            can be an attribute name (str) or a function (callable) that returns the file path.

    Returns:
        bool: True if all conditions are met (use_flag is True, config is not None,
              and the resolved file path is not empty), otherwise False.
    """

    if not use_flag or config is None:
        return False

    filepath = (
        filepath_resolver(config)
        if callable(filepath_resolver)
        else getattr(config, filepath_resolver, "")
    )
    return bool(filepath)


def open_and_return_datapipes(
    configuration_filename: str,
    use_gsp: bool = True,
    use_nwp: bool = True,
    use_pv: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_topo: bool = True,
    use_sensor: bool = True,
    use_wind: bool = True,
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
        use_wind: Whether to use wind data or not

    Returns:
        List of datapipes corresponding to the datapipes to open
    """
    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Check which modalities to use
    conf_in = configuration.input_data
    use_nwp = (
        use_nwp
        and (conf_in.nwp is not None)
        and len(conf_in.nwp) != 0
        and all(v.nwp_zarr_path != "" for _, v in conf_in.nwp.items())
    )

    use_pv = is_config_and_path_valid(
        use_pv,
        conf_in.pv,
        lambda config: config.pv_files_groups[0].pv_filename if config.pv_files_groups else "",
    )
    use_sat = is_config_and_path_valid(use_sat, conf_in.satellite, "satellite_zarr_path")
    use_hrv = is_config_and_path_valid(use_hrv, conf_in.hrvsatellite, "hrvsatellite_zarr_path")
    use_topo = is_config_and_path_valid(use_topo, conf_in.topographic, "topographic_filename")
    use_gsp = is_config_and_path_valid(use_gsp, conf_in.gsp, "gsp_zarr_path")
    use_sensor = is_config_and_path_valid(use_sensor, conf_in.sensor, "sensor_filename")
    use_wind = is_config_and_path_valid(
        use_wind,
        conf_in.wind,
        lambda config: (
            config.wind_files_groups[0].wind_filename if config.wind_files_groups else ""
        ),
    )

    logger.debug(
        f"GSP: {use_gsp} NWP: {use_nwp} Sat: {use_sat},"
        f" HRV: {use_hrv} PV: {use_pv} Topo: {use_topo}"
        f" Sensor: {use_sensor} Wind: {use_wind}"
    )

    used_datapipes = {}

    used_datapipes["config"] = configuration

    # Load GSP national data
    if use_gsp:
        logger.debug("Opening GSP Data")
        gsp_datapipe = OpenGSP(
            gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=minutes(configuration.input_data.gsp.time_resolution_minutes),
            history_duration=minutes(configuration.input_data.gsp.history_minutes),
        )

        used_datapipes["gsp"] = gsp_datapipe

    # Load NWP data
    if use_nwp:
        logger.debug("Opening NWP Data")
        used_datapipes["nwp"] = {}
        for nwp_source, nwp_conf in conf_in.nwp.items():
            used_datapipes["nwp"][nwp_source] = (
                OpenNWP(
                    nwp_conf.nwp_zarr_path,
                    provider=nwp_conf.nwp_provider,
                )
                .filter_channels(
                    nwp_conf.nwp_channels,
                    provider=nwp_conf.nwp_provider,
                )
                .add_t0_idx_and_sample_period_duration(
                    sample_period_duration=minutes(nwp_conf.time_resolution_minutes),
                    history_duration=minutes(nwp_conf.history_minutes),
                )
            )

    if use_sat:
        logger.debug("Opening Satellite Data")
        sat_datapipe = (
            OpenSatellite(configuration.input_data.satellite.satellite_zarr_path)
            .filter_channels(configuration.input_data.satellite.satellite_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=minutes(
                    configuration.input_data.satellite.time_resolution_minutes
                ),
                history_duration=minutes(configuration.input_data.satellite.history_minutes),
            )
        )

        used_datapipes["sat"] = sat_datapipe

    if use_hrv:
        logger.debug("Opening HRV Satellite Data")
        sat_hrv_datapipe = OpenSatellite(
            configuration.input_data.hrvsatellite.hrvsatellite_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=minutes(
                configuration.input_data.hrvsatellite.time_resolution_minutes
            ),
            history_duration=minutes(configuration.input_data.hrvsatellite.history_minutes),
        )

        used_datapipes["hrv"] = sat_hrv_datapipe

    if use_pv:
        logger.debug("Opening PV")
        pv_datapipe = OpenPVFromNetCDF(
            pv=configuration.input_data.pv
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=minutes(configuration.input_data.pv.time_resolution_minutes),
            history_duration=minutes(configuration.input_data.pv.history_minutes),
        )

        used_datapipes["pv"] = pv_datapipe

    if use_wind:
        logger.debug("Opening Wind")
        wind_datapipe = OpenWindFromNetCDF(
            wind=configuration.input_data.wind
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=minutes(configuration.input_data.wind.time_resolution_minutes),
            history_duration=minutes(configuration.input_data.wind.history_minutes),
        )

        used_datapipes["wind"] = wind_datapipe

    if use_sensor:
        logger.debug("Opening Sensor Data")
        sensor_datapipe = OpenMeteomaticsFromZarr(
            configuration.input_data.sensor
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=minutes(configuration.input_data.sensor.time_resolution_minutes),
            history_duration=minutes(configuration.input_data.sensor.history_minutes),
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
    used_datapipes = flatten_nwp_source_dict(used_datapipes)

    for key, datapipe in used_datapipes.items():
        if "topo" in key:
            continue
        if key_for_t0 in key:
            forked_datapipes = datapipe.fork(3, buffer_size=100)
            t0_datapipe = forked_datapipes[2]
        else:
            forked_datapipes = datapipe.fork(2, buffer_size=100)
        datapipes_to_return[key] = forked_datapipes[0]
        if key.startswith("nwp/"):
            nwp_source = key.removeprefix("nwp/")
            time_periods_datapipe = forked_datapipes[1].find_contiguous_t0_time_periods(
                sample_period_duration=timedelta(hours=3),  # Init times are 3 hours apart
                history_duration=minutes(configuration.input_data.nwp[nwp_source].history_minutes),
                forecast_duration=minutes(
                    configuration.input_data.nwp[nwp_source].forecast_minutes
                ),
                time_dim="init_time_utc",
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "sat" == key:
            time_periods_datapipe = forked_datapipes[1].find_contiguous_t0_time_periods(
                sample_period_duration=minutes(
                    configuration.input_data.satellite.time_resolution_minutes
                ),
                history_duration=minutes(configuration.input_data.satellite.history_minutes),
                forecast_duration=minutes(0),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "hrv" == key:
            time_periods_datapipe = forked_datapipes[1].find_contiguous_t0_time_periods(
                sample_period_duration=minutes(
                    configuration.input_data.hrvsatellite.time_resolution_minutes
                ),
                history_duration=minutes(configuration.input_data.hrvsatellite.history_minutes),
                forecast_duration=minutes(0),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "pv" == key:
            time_periods_datapipe = forked_datapipes[1].find_contiguous_t0_time_periods(
                sample_period_duration=minutes(configuration.input_data.pv.time_resolution_minutes),
                history_duration=minutes(configuration.input_data.pv.history_minutes),
                forecast_duration=minutes(configuration.input_data.pv.forecast_minutes),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)
        if "wind" == key:
            time_periods_datapipe = forked_datapipes[1].find_contiguous_t0_time_periods(
                sample_period_duration=minutes(
                    configuration.input_data.wind.time_resolution_minutes
                ),
                history_duration=minutes(configuration.input_data.wind.history_minutes),
                forecast_duration=minutes(configuration.input_data.wind.forecast_minutes),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)
        if "gsp" == key:
            time_periods_datapipe = forked_datapipes[1].find_contiguous_t0_time_periods(
                sample_period_duration=minutes(
                    configuration.input_data.gsp.time_resolution_minutes
                ),
                history_duration=minutes(configuration.input_data.gsp.history_minutes),
                forecast_duration=minutes(configuration.input_data.gsp.forecast_minutes),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

        if "sensor" == key:
            time_periods_datapipe = forked_datapipes[1].find_contiguous_t0_time_periods(
                sample_period_duration=minutes(
                    configuration.input_data.sensor.time_resolution_minutes
                ),
                history_duration=minutes(configuration.input_data.sensor.history_minutes),
                forecast_duration=minutes(configuration.input_data.sensor.forecast_minutes),
            )
            datapipes_for_time_periods.append(time_periods_datapipe)

    # Now have the forked ones
    # find joint overlapping timer periods
    logger.debug("Getting joint time periods")
    overlapping_datapipe = datapipes_for_time_periods[0].filter_to_overlapping_time_periods(
        secondary_datapipes=datapipes_for_time_periods[1:],
    )

    # select time periods
    t0_datapipe = t0_datapipe.filter_time_periods(time_periods=overlapping_datapipe)

    num_t0_datapipes = len(datapipes_to_return.keys())  # One for each input
    t0_datapipes = t0_datapipe.pick_t0_times().fork(num_t0_datapipes, buffer_size=100)

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


def normalize_wind(x):
    """
    Normalize the wind data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return (x / x.nominal_capacity_mwp).clip(None, 5)


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

        Returns:model_validator
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


def fill_nans_in_arrays(
    batch: NumpyBatch, alert=True, _filled_keys=set(), _key_prefix=""
) -> NumpyBatch:
    """Fills all NaN values in each np.ndarray in the batch dictionary with zeros.

    Operation is performed in-place on the batch.
    """
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            if np.isnan(v).any():
                _filled_keys.update({f"{_key_prefix}{k}"})
                batch[k] = np.nan_to_num(v, copy=True, nan=0.0)

        # Recursion is included to reach NWP arrays in subdict
        elif isinstance(v, dict):
            fill_nans_in_arrays(
                v, alert=False, _filled_keys=_filled_keys, _key_prefix=f"{_key_prefix}{k}/"
            )

    if alert and len(_filled_keys) > 0:
        logger.info(f"Filled NaNs with zeros - {_filled_keys}")
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
        assert len(self.keys_left) == 0, self.keys_left


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
        use_wind=True,
    )

    config: Configuration = datapipes_dict["config"]

    if production:
        datapipes_dict["gsp"] = OpenGSPFromDatabase().add_t0_idx_and_sample_period_duration(
            sample_period_duration=minutes(config.input_data.gsp.time_resolution_minutes),
            history_duration=minutes(config.input_data.gsp.history_minutes),
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
    # data first and this is carried through
    preferred_order_of_keys = [
        "gsp",
        "pv",
        "wind",
        "sensor",
    ]

    core_key = next(filter(lambda key: key in datapipes_dict, preferred_order_of_keys))

    if "gsp" in datapipes_dict:
        datapipes_dict["gsp"] = datapipes_dict["gsp"].map(gsp_drop_national)

    if (start_time is not None) or (end_time is not None):
        datapipes_dict[core_key] = datapipes_dict[core_key].filter_times(start_time, end_time)

    # Get overlapping time periods
    location_pipe, t0_datapipe = create_t0_and_loc_datapipes(
        datapipes_dict,
        configuration=config,
        key_for_t0=core_key,
        shuffle=True,
    )

    return location_pipe, t0_datapipe


def minutes(num_mins: int):
    """Timedelta of a number of minutes.

    Args:
        num_mins: Minutes timedelta.
    """
    return timedelta(minutes=num_mins)


def minutes_list_to_timedeltas(list_ints):
    """Utility function to convert list of dropout timedelta minutes ints to list of timedeltas"""
    if list_ints is None:
        return list_ints
    else:
        return [minutes(m) for m in list_ints]


def slice_datapipes_by_time(
    datapipes_dict: Dict,
    t0_datapipe: IterDataPipe,
    configuration: Configuration,
    production: bool = False,
) -> None:
    """
    Modifies a dictionary of datapipes in-place to yield samples for given times t0.

    Note that where dropout is mentioned here, this is only applied if production=False.

    The NWP data* will may include dropout to a earlier init time depending on the config.

    The satellite data* may also include dropout where some timestamps are missing and replaced with
    all NaNs, depending on the dropout settings in the config.

    The HRV data* is similar to the satellite data and if both are included they drop out
    simulataneously.

    The GSP data is split into "gsp" and "gsp_future" keys. Depending on the config, the most recent
    data may be dropped out and replaced with NaNs

    The PV data* is also split it "pv" and "pv_future" keys. Depending on the config, the most
    recent data may be dropped out and replaced with NaNs. Additionally, the PV systems may be
    independenly dropped out rather than a constant dropout time being used across all systems.

    * if included

    n.b. HRV is included in this function, but is not yet in the rest of the pvnet pipeline.
    This is mostly for demonstration purposes of how concurrent dropout might be applied to HRV and
    non-HRV satellite.

    Args:
        datapipes_dict: Dictionary of used datapipes and t0 ones
        t0_datapipe: Datapipe which yields t0 times for sample
        configuration: Configuration object.
        production: Whether constucting pipeline for production inference. No dropout is used if
            True.

    """

    conf_in = configuration.input_data

    # Use DatapipeKeyForker to avoid forking t0_datapipe too many times, or leaving any forks unused
    fork_keys = {k for k in datapipes_dict.keys() if k not in ["topo", "nwp"]}
    if "nwp" in datapipes_dict:  # nwp is nested so treat separately
        fork_keys.update({f"nwp/{k}" for k in datapipes_dict["nwp"].keys()})

    get_t0_datapipe = DatapipeKeyForker(fork_keys, t0_datapipe)

    # Satelite and HRV satellite should drop out simultaneously when they dropout
    if "sat" in datapipes_dict or "hrv" in datapipes_dict:
        if "sat" in datapipes_dict and "hrv" in datapipes_dict:
            logger.warn(
                "Both sat and hrv in data sources. But dropout values will only be taken from sat "
                "and apllied to both"
            )

        if "sat" in datapipes_dict:
            conf_sathrv = conf_in.satellite
        else:
            conf_sathrv = conf_in.hrvsatellite

        dropout_timedeltas = minutes_list_to_timedeltas(conf_sathrv.dropout_timedeltas_minutes)

        sat_and_hrv_dropout_kwargs = dict(
            # Satellite is either 30 minutes or 60 minutes delayed in production.
            # Match during training
            dropout_timedeltas=dropout_timedeltas,
            dropout_frac=0 if production else conf_sathrv.dropout_fraction,
        )

        sat_delay = minutes(-conf_sathrv.live_delay_minutes)

    if "nwp" in datapipes_dict:
        # NWP is nested in the dict
        for nwp_key, dp in datapipes_dict["nwp"].items():
            dropout_timedeltas = minutes_list_to_timedeltas(
                conf_in.nwp[nwp_key].dropout_timedeltas_minutes
            )

            datapipes_dict["nwp"][nwp_key] = dp.select_time_slice_nwp(
                t0_datapipe=get_t0_datapipe(f"nwp/{nwp_key}"),
                sample_period_duration=minutes(conf_in.nwp[nwp_key].time_resolution_minutes),
                history_duration=minutes(conf_in.nwp[nwp_key].history_minutes),
                forecast_duration=minutes(conf_in.nwp[nwp_key].forecast_minutes),
                dropout_timedeltas=dropout_timedeltas,
                dropout_frac=0 if production else conf_in.nwp[nwp_key].dropout_fraction,
                accum_channels=conf_in.nwp[nwp_key].nwp_accum_channels,
            )

    if "sat" in datapipes_dict:
        # Take time slices of sat data
        datapipes_dict["sat"] = datapipes_dict["sat"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.satellite.time_resolution_minutes),
            interval_start=minutes(-conf_in.satellite.history_minutes),
            interval_end=sat_delay,
            fill_selection=production,
            max_steps_gap=2,
        )

        # Generate randomly sampled dropout times
        sat_dropout_time_datapipe = get_t0_datapipe("sat").draw_dropout_time(
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
            hrv_dropout_time_datapipe = get_t0_datapipe(None).draw_dropout_time(
                **sat_and_hrv_dropout_kwargs
            )

        datapipes_dict["hrv"] = datapipes_dict["hrv"].select_time_slice(
            t0_datapipe=get_t0_datapipe("hrv"),
            sample_period_duration=minutes(conf_in.hrvsatellite.time_resolution_minutes),
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
            sample_period_duration=minutes(conf_in.pv.time_resolution_minutes),
            interval_start=minutes(5),
            interval_end=minutes(conf_in.pv.forecast_minutes),
            fill_selection=production,
        )

        datapipes_dict["pv"] = datapipes_dict["pv"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.pv.time_resolution_minutes),
            interval_start=minutes(-conf_in.pv.history_minutes),
            interval_end=minutes(0),
            fill_selection=production,
        )

        # Dropout on the PV, but not the future PV

        dropout_timedeltas = minutes_list_to_timedeltas(conf_in.pv.dropout_timedeltas_minutes)

        pv_dropout_time_datapipe = get_t0_datapipe("pv").draw_dropout_time(
            dropout_timedeltas=dropout_timedeltas,
            dropout_frac=0 if production else conf_in.pv.dropout_fraction,
        )

        datapipes_dict["pv"] = datapipes_dict["pv"].apply_dropout_time(
            dropout_time_datapipe=pv_dropout_time_datapipe,
        )

        # Apply extra PV dropout using different delays per system and dropping out
        # entire PV systems independently
        if not production:
            system_dropout_timedeltas = minutes_list_to_timedeltas(
                conf_in.pv.system_dropout_timedeltas_minutes
            )

            datapipes_dict["pv"].apply_pv_dropout(
                min_frac=conf_in.pv.system_dropout_fraction_min,
                max_frac=conf_in.pv.system_dropout_fraction_max,
                system_dropout_timedeltas=system_dropout_timedeltas,
            )

    if "wind" in datapipes_dict:
        datapipes_dict["wind"], dp = datapipes_dict["wind"].fork(2, buffer_size=5)

        datapipes_dict["wind_future"] = dp.select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.wind.time_resolution_minutes),
            interval_start=minutes(15),
            interval_end=minutes(conf_in.wind.forecast_minutes),
            fill_selection=production,
        )

        datapipes_dict["wind"] = datapipes_dict["wind"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.wind.time_resolution_minutes),
            interval_start=minutes(-conf_in.wind.history_minutes),
            interval_end=minutes(0),
            fill_selection=production,
        )

        dropout_timedeltas = minutes_list_to_timedeltas(conf_in.wind.dropout_timedeltas_minutes)

        # Dropout on the Wind, but not the future Wind
        wind_dropout_time_datapipe = get_t0_datapipe("wind").draw_dropout_time(
            # All Wind data could be delayed by up to 30 minutes
            # (this does not stem from production - just setting for now)
            dropout_timedeltas=dropout_timedeltas,
            dropout_frac=0 if production else conf_in.wind.dropout_fraction,
        )

        datapipes_dict["wind"] = datapipes_dict["wind"].apply_dropout_time(
            dropout_time_datapipe=wind_dropout_time_datapipe,
        )

    if "sensor" in datapipes_dict:
        datapipes_dict["sensor"], dp = datapipes_dict["sensor"].fork(2, buffer_size=5)

        datapipes_dict["sensor_future"] = dp.select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.sensor.time_resolution_minutes),
            interval_start=minutes(15),
            interval_end=minutes(conf_in.sensor.forecast_minutes),
            fill_selection=production,
        )

        datapipes_dict["sensor"] = datapipes_dict["sensor"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.sensor.time_resolution_minutes),
            interval_start=minutes(-conf_in.sensor.history_minutes),
            interval_end=minutes(conf_in.sensor.forecast_minutes),
            fill_selection=production,
        )

        sensor_dropout_time_datapipe = get_t0_datapipe("sensor").draw_dropout_time(
            dropout_timedeltas=0,
            dropout_frac=0,
        )

        datapipes_dict["sensor"] = datapipes_dict["sensor"].apply_dropout_time(
            dropout_time_datapipe=sensor_dropout_time_datapipe,
        )

    if "gsp" in datapipes_dict:
        datapipes_dict["gsp"], dp = datapipes_dict["gsp"].fork(2, buffer_size=5)

        datapipes_dict["gsp_future"] = dp.select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.gsp.time_resolution_minutes),
            interval_start=minutes(30),
            interval_end=minutes(conf_in.gsp.forecast_minutes),
            fill_selection=production,
        )

        datapipes_dict["gsp"] = datapipes_dict["gsp"].select_time_slice(
            t0_datapipe=get_t0_datapipe(None),
            sample_period_duration=minutes(conf_in.gsp.time_resolution_minutes),
            interval_start=-minutes(conf_in.gsp.history_minutes),
            interval_end=minutes(0),
            fill_selection=production,
        )

        # Dropout on the GSP, but not the future GSP
        dropout_timedeltas = minutes_list_to_timedeltas(conf_in.gsp.dropout_timedeltas_minutes)

        gsp_dropout_time_datapipe = get_t0_datapipe("gsp").draw_dropout_time(
            dropout_timedeltas=dropout_timedeltas,
            dropout_frac=0 if production else conf_in.gsp.dropout_fraction,
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
                    sat_data_one_step = batch[BatchKey.satellite_actual][t]
                else:
                    sat_data_one_step = batch[BatchKey.satellite_actual][:, t]

                nans = np.isnan(sat_data_one_step)

                if np.any(nans):
                    percent_nans = np.mean(nans) * 100

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
        if key.startswith("nwp/"):
            nwp_source = key.removeprefix("nwp/")
            datapipes_to_return[key] = datapipe.select_time_slice_nwp(
                t0_datapipe=used_datapipes[key + "_t0"],
                sample_period_duration=timedelta(hours=1),
                history_duration=minutes(configuration.input_data.nwp[nwp_source].history_minutes),
                forecast_duration=minutes(
                    configuration.input_data.nwp[nwp_source].forecast_minutes
                ),
            )

        if "sat" == key:
            datapipes_to_return[key] = datapipe.select_time_slice(
                t0_datapipe=used_datapipes[key + "_t0"],
                history_duration=minutes(configuration.input_data.satellite.history_minutes),
                forecast_duration=minutes(0),
                sample_period_duration=minutes(
                    configuration.input_data.satellite.time_resolution_minutes
                ),
            )

        if "hrv" == key:
            datapipes_to_return[key] = datapipe.select_time_slice(
                t0_datapipe=used_datapipes[key + "_t0"],
                history_duration=minutes(configuration.input_data.hrvsatellite.history_minutes),
                forecast_duration=minutes(0),
                sample_period_duration=minutes(
                    configuration.input_data.hrvsatellite.time_resolution_minutes
                ),
            )

        if "pv" == key:
            pv_1, pv_2 = used_datapipes[key + "_t0"].fork(2)
            pv_dp1, pv_dp2 = datapipe.fork(2)
            datapipes_to_return[key] = pv_dp1.select_time_slice(
                t0_datapipe=pv_1,
                history_duration=minutes(configuration.input_data.pv.history_minutes),
                forecast_duration=minutes(0),
                sample_period_duration=minutes(configuration.input_data.pv.time_resolution_minutes),
            )
            datapipes_to_return[key + "_future"] = pv_dp2.select_time_slice(
                t0_datapipe=pv_2,
                history_duration=minutes(0),
                forecast_duration=minutes(configuration.input_data.pv.forecast_minutes),
                sample_period_duration=minutes(configuration.input_data.pv.time_resolution_minutes),
            )

        if "wind" == key:
            wind_1, wind_2 = used_datapipes[key + "_t0"].fork(2)
            wind_dp1, wind_dp2 = datapipe.fork(2)
            datapipes_to_return[key] = wind_dp1.select_time_slice(
                t0_datapipe=wind_1,
                history_duration=minutes(configuration.input_data.wind.history_minutes),
                forecast_duration=minutes(0),
                sample_period_duration=minutes(
                    configuration.input_data.wind.time_resolution_minutes
                ),
            )
            datapipes_to_return[key + "_future"] = wind_dp2.select_time_slice(
                t0_datapipe=wind_2,
                history_duration=minutes(0),
                forecast_duration=minutes(configuration.input_data.wind.forecast_minutes),
                sample_period_duration=minutes(
                    configuration.input_data.wind.time_resolution_minutes
                ),
            )

        if "sensor" == key:
            sensor_1, sensor_2 = used_datapipes[key + "_t0"].fork(2)
            sensor_dp1, sensor_dp2 = datapipe.fork(2)
            datapipes_to_return[key] = sensor_dp1.select_time_slice(
                t0_datapipe=sensor_1,
                history_duration=minutes(configuration.input_data.sensor.history_minutes),
                forecast_duration=minutes(0),
                sample_period_duration=minutes(
                    configuration.input_data.sensor.time_resolution_minutes
                ),
            )
            datapipes_to_return[key + "_future"] = sensor_dp2.select_time_slice(
                t0_datapipe=sensor_2,
                history_duration=minutes(0),
                forecast_duration=minutes(configuration.input_data.sensor.forecast_minutes),
                sample_period_duration=minutes(
                    configuration.input_data.sensor.time_resolution_minutes
                ),
            )

        if "gsp" == key:
            gsp_1, gsp_2 = used_datapipes[key + "_t0"].fork(2)
            gsp_dp1, gsp_dp2 = datapipe.fork(2)
            datapipes_to_return[key] = gsp_dp1.select_time_slice(
                t0_datapipe=gsp_1,
                history_duration=minutes(configuration.input_data.gsp.history_minutes),
                forecast_duration=minutes(0),
                sample_period_duration=minutes(
                    configuration.input_data.gsp.time_resolution_minutes
                ),
            )
            datapipes_to_return[key + "_future"] = gsp_dp2.select_time_slice(
                t0_datapipe=gsp_2,
                history_duration=minutes(0),
                forecast_duration=minutes(configuration.input_data.gsp.forecast_minutes),
                sample_period_duration=minutes(
                    configuration.input_data.gsp.time_resolution_minutes
                ),
            )
    if "topo" in used_datapipes.keys():
        datapipes_to_return["topo"] = used_datapipes["topo"]
    datapipes_to_return["config"] = configuration
    return datapipes_to_return


def create_valid_t0_periods_datapipe(
    datapipes_dict: dict,
    configuration: Configuration,
    key_for_t0: str = "gsp",
):
    """Create datapipe yielding t0 periods which are valid for the input data sources.

    Args:
        datapipes_dict: Dictionary of datapipes of input sources for which we want to select
            appropriate location and times.
        configuration: Configuration object for inputs.
        key_for_t0: Key to use for the t0 datapipe. Must be "gsp" or "pv".
    """
    assert key_for_t0 in datapipes_dict
    assert key_for_t0 in [
        "gsp",
        "pv",
        "wind",
        "sensor",
    ]

    contiguous_time_datapipes = []  # Used to store contiguous time periods from each data source

    datapipes_dict[key_for_t0], key_datapipe = datapipes_dict[key_for_t0].fork(2, buffer_size=5)

    for key in datapipes_dict.keys():
        if key in ["topo"]:
            continue

        elif key == "nwp":
            for nwp_key in datapipes_dict["nwp"].keys():
                # NWPs are nested since there can be multiple NWP sources
                datapipes_dict["nwp"][nwp_key], datapipe_copy = datapipes_dict["nwp"][nwp_key].fork(
                    2, buffer_size=5
                )

                # Different config setting per NWP source
                nwp_conf = configuration.input_data.nwp[nwp_key]

                if nwp_conf.dropout_timedeltas_minutes is None:
                    max_dropout = minutes(0)
                else:
                    max_dropout = minutes(int(np.max(np.abs(nwp_conf.dropout_timedeltas_minutes))))

                if nwp_conf.max_staleness_minutes is None:
                    max_staleness = None
                else:
                    max_staleness = minutes(nwp_conf.max_staleness_minutes)

                # If we are diffing some the accumulatd channels, we can't use the last time stamp
                # of the NWP forecast
                if len(nwp_conf.nwp_accum_channels) > 0:
                    end_buffer = minutes(60)
                else:
                    end_buffer = minutes(0)

                # NWP is a forecast product so gets its own contiguous function
                time_periods = datapipe_copy.find_contiguous_t0_time_periods_nwp(
                    history_duration=minutes(nwp_conf.history_minutes),
                    forecast_duration=minutes(nwp_conf.forecast_minutes),
                    max_staleness=max_staleness,
                    max_dropout=max_dropout,
                    time_dim="init_time_utc",
                    end_buffer=end_buffer,
                )

                contiguous_time_datapipes.append(time_periods)

        else:
            if key == "sat":
                sample_frequency = configuration.input_data.satellite.time_resolution_minutes
                history_duration = configuration.input_data.satellite.history_minutes
                forecast_duration = 0
                time_dim = "time_utc"

            elif key == "hrv":
                sample_frequency = configuration.input_data.hrvsatellite.time_resolution_minutes
                history_duration = configuration.input_data.hrvsatellite.history_minutes
                forecast_duration = 0
                time_dim = "time_utc"

            elif key == "pv":
                sample_frequency = configuration.input_data.pv.time_resolution_minutes
                history_duration = configuration.input_data.pv.history_minutes
                forecast_duration = configuration.input_data.pv.forecast_minutes
                time_dim = "time_utc"

            elif key == "wind":
                sample_frequency = configuration.input_data.wind.time_resolution_minutes
                history_duration = configuration.input_data.wind.history_minutes
                forecast_duration = configuration.input_data.wind.forecast_minutes
                time_dim = "time_utc"

            elif key == "sensor":
                sample_frequency = configuration.input_data.sensor.time_resolution_minutes
                history_duration = configuration.input_data.sensor.history_minutes
                forecast_duration = configuration.input_data.sensor.forecast_minutes
                time_dim = "time_utc"

            elif key == "gsp":
                sample_frequency = configuration.input_data.gsp.time_resolution_minutes
                history_duration = configuration.input_data.gsp.history_minutes
                forecast_duration = configuration.input_data.gsp.forecast_minutes
                time_dim = "time_utc"

            else:
                raise ValueError(f"Unexpected key: {key}")

            datapipes_dict[key], datapipe_copy = datapipes_dict[key].fork(2, buffer_size=5)

            time_periods = datapipe_copy.find_contiguous_t0_time_periods(
                sample_period_duration=minutes(sample_frequency),
                history_duration=minutes(history_duration),
                forecast_duration=minutes(forecast_duration),
                time_dim=time_dim,
            )

            contiguous_time_datapipes.append(time_periods)

    # Find joint overlapping contiguous time periods
    if len(contiguous_time_datapipes) > 1:
        logger.debug("Getting joint time periods")
        overlapping_datapipe = contiguous_time_datapipes[0].filter_to_overlapping_time_periods(
            secondary_datapipes=contiguous_time_datapipes[1:],
        )
    else:
        logger.debug("Skipping getting joint time periods")
        overlapping_datapipe = contiguous_time_datapipes[0]

    # Select time periods and set length
    valid_t0_periods_datapipe = key_datapipe.filter_time_periods(time_periods=overlapping_datapipe)

    return valid_t0_periods_datapipe


def create_t0_and_loc_datapipes(
    datapipes_dict: dict,
    configuration: Configuration,
    key_for_t0: str = "gsp",
    shuffle: bool = True,
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

    Returns:
        location datapipe, t0 datapipe
    """

    valid_t0_periods_datapipe = create_valid_t0_periods_datapipe(
        datapipes_dict,
        configuration,
        key_for_t0,
    )

    t0_loc_datapipe = valid_t0_periods_datapipe.pick_locs_and_t0s(return_all=True, shuffle=shuffle)

    location_pipe, t0_datapipe = t0_loc_datapipe.unzip(sequence_length=2)

    return location_pipe, t0_datapipe


def potentially_coarsen(xr_data: xr.Dataset, coarsen_to_deg: float = 0.1):
    """
    Coarsen the data, change the latitude longitude grid

    Args:
        xr_data: xarray dataset
        coarsen_to_deg: Coarsen to this degree in lat and lon
    """
    if "latitude" in xr_data.coords and "longitude" in xr_data.coords:
        step = np.abs(xr_data.latitude.values[1] - xr_data.latitude.values[0])
        step = np.round(step, 4)
        coarsen_factor = int(coarsen_to_deg / step)
        if coarsen_factor > 1:
            xr_data = xr_data.coarsen(
                latitude=coarsen_factor, longitude=coarsen_factor, boundary="pad", coord_func="min"
            ).mean()
            # we use the coord_func "min", as the using "mean"
            # results in with some fractions of coordinates (sometimes),
            # e.g. mean of 0.05, 0.1, 0.15, 0.2 is 0.125,
    return xr_data
