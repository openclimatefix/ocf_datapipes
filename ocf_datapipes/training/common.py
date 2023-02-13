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

    Returns:
        List of datapipes corresponding to the datapipes to open
    """
    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Check which modalities to use
    if use_nwp:
        use_nwp = True if configuration.input_data.nwp.nwp_zarr_path != "" else False
    if use_pv:
        use_pv = True if configuration.input_data.pv.pv_files_groups[0].pv_filename != "" else False
    if use_sat:
        use_sat = True if configuration.input_data.satellite.satellite_zarr_path != "" else False
    if use_hrv:
        use_hrv = (
            True if configuration.input_data.hrvsatellite.hrvsatellite_zarr_path != "" else False
        )
    if use_topo:
        use_topo = (
            True if configuration.input_data.topographic.topographic_filename != "" else False
        )
    if use_gsp:
        use_gsp = True if configuration.input_data.gsp.gsp_zarr_path != "" else False
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
            OpenSatellite(configuration.input_data.satellite.satellite_zarr_path)
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
            forked_datapipes = datapipe.fork(3, buffer_size=5)
            t0_datapipe = forked_datapipes[2]
        else:
            forked_datapipes = datapipe.fork(2, buffer_size=5)
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
        num_t0_datapipes, buffer_size=5
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
