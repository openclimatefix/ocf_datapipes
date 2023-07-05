"""Common functionality for datapipes"""
import logging
from datetime import timedelta

from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import (
    OpenConfiguration,
    OpenGSP,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenSatellite,
    OpenTopography,
)

logger = logging.getLogger(__name__)


def open_and_return_datapipes(
    configuration_filename: str,
    use_gsp: bool = True,
    use_nwp: bool = True,
    use_pv: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_topo: bool = True,
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
        production: bool if this is for production or not

    Returns:
        List of datapipes corresponding to the datapipes to open
    """
    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Check which modalities to use
    conf_in = configuration.input_data
    use_nwp = use_nwp and (conf_in.nwp.nwp_zarr_path != "")
    use_pv = use_pv and (conf_in.pv.pv_files_groups[0].pv_filename != "")
    use_sat = use_sat and (conf_in.satellite.satellite_zarr_path != "")
    use_hrv = use_hrv and (conf_in.hrvsatellite.hrvsatellite_zarr_path != "")
    use_topo = use_topo and (conf_in.topographic.topographic_filename != "")
    use_gsp = use_gsp and (conf_in.gsp.gsp_zarr_path != "")

    logger.debug(
        f"GSP: {use_gsp} NWP: {use_nwp} Sat: {use_sat},"
        f" HRV: {use_hrv} PV: {use_pv} Topo: {use_topo}"
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
            OpenNWP(configuration.input_data.nwp.nwp_zarr_path)
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

    # Readd config for later
    datapipes_to_return["config"] = configuration
    if "topo" in used_datapipes.keys():
        datapipes_to_return["topo"] = used_datapipes["topo"]
    return datapipes_to_return


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
    nwp_max_t0_offset: timedelta = timedelta(minutes=0),
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
        nwp_max_t0_offset: If using dropout on NWP, sometimes we have to go back to previous NWP
            init time. In order to accomodat for this possibility in selecting times, set
            `nwp_max_t0_offset` as the max NWP dropout delay you plan to use.

    Returns:
        location datapipe, t0 datapipe

    """
    assert key_for_t0 in datapipes_dict
    assert key_for_t0 in ["gsp", "pv"]

    contiguous_time_datapipes = []  # Used to store contiguous time periods from each data source

    datapipes_dict[key_for_t0], key_datapipe = datapipes_dict[key_for_t0].fork(2, buffer_size=5)

    for key in datapipes_dict.keys():
        if key in ["topo"]:
            continue

        elif key == "nwp":
            sample_frequency = 180  # Init times are 3 hours apart
            history_duration = configuration.input_data.nwp.history_minutes
            forecast_duration = configuration.input_data.nwp.forecast_minutes
            time_dim = "init_time_utc"
            max_t0_offset = nwp_max_t0_offset

        elif key == "sat":
            sample_frequency = 5
            history_duration = configuration.input_data.satellite.history_minutes
            forecast_duration = 0
            time_dim = "time_utc"
            max_t0_offset = timedelta(minutes=0)

        elif key == "hrv":
            sample_frequency = 5
            history_duration = configuration.input_data.hrvsatellite.history_minutes
            forecast_duration = 0
            time_dim = "time_utc"
            max_t0_offset = timedelta(minutes=0)

        elif key == "pv":
            sample_frequency = 5
            history_duration = configuration.input_data.pv.history_minutes
            forecast_duration = configuration.input_data.pv.forecast_minutes
            time_dim = "time_utc"
            max_t0_offset = timedelta(minutes=0)

        elif key == "gsp":
            sample_frequency = 30
            history_duration = configuration.input_data.gsp.history_minutes
            forecast_duration = configuration.input_data.gsp.forecast_minutes
            time_dim = "time_utc"
            max_t0_offset = timedelta(minutes=0)

        else:
            raise ValueError(f"Unexpected key: {key}")

        datapipes_dict[key], datapipe_copy = datapipes_dict[key].fork(2, buffer_size=5)

        time_periods = datapipe_copy.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=sample_frequency),
            history_duration=timedelta(minutes=history_duration),
            forecast_duration=timedelta(minutes=forecast_duration),
            time_dim=time_dim,
            max_t0_offset=max_t0_offset,
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
