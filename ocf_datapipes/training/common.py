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
logger.setLevel(logging.DEBUG)


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
    # Load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))
    
    conf_in = configuration.input_data
    # Filter modalities to those with filepaths
    use_nwp = use_nwp and (conf_in.nwp.nwp_zarr_path != "")
    use_pv = use_pv and (conf_in.pv.pv_files_groups[0].pv_filename != "")
    use_sat = use_sat and (conf_in.satellite.satellite_zarr_path != "") 
    use_hrv =  use_hrv and (conf_in.hrvsatellite.hrvsatellite_zarr_path != "")
    use_topo = use_topo and (conf_in.topographic.topographic_filename != "")
    use_gsp = use_gsp and (conf_in.gsp.gsp_zarr_path != "")
    
    logger.debug(
        f"GSP: {use_gsp} NWP: {use_nwp} Sat: {use_sat},"
        f" HRV: {use_hrv} PV: {use_pv} Topo: {use_topo}"
    )
    
    # Initiate pipeline dict
    used_datapipes = {"config":configuration}

    # Load GSP national data
    if use_gsp:
        logger.debug("Opening GSP Data")
        gsp_datapipe = OpenGSP(
            gsp_pv_power_zarr_path=conf_in.gsp.gsp_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=conf_in.gsp.history_minutes),
        )

        used_datapipes["gsp"] = gsp_datapipe

    # Load NWP data
    if use_nwp:
        logger.debug("Opening NWP Data")
        nwp_datapipe = (
            OpenNWP(conf_in.nwp.nwp_zarr_path)
            .select_channels(conf_in.nwp.nwp_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(hours=1),
                history_duration=timedelta(minutes=conf_in.nwp.history_minutes),
            )
        )

        used_datapipes["nwp"] = nwp_datapipe

    if use_sat:
        logger.debug("Opening Satellite Data")
        sat_datapipe = (
            OpenSatellite(conf_in.satellite.satellite_zarr_path)
            .select_channels(conf_in.satellite.satellite_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(
                    minutes=conf_in.satellite.history_minutes
                ),
            )
        )

        used_datapipes["sat"] = sat_datapipe

    if use_hrv:
        logger.debug("Opening HRV Satellite Data")
        sat_hrv_datapipe = OpenSatellite(
            conf_in.hrvsatellite.hrvsatellite_zarr_path
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(
                minutes=conf_in.hrvsatellite.history_minutes
            ),
        )

        used_datapipes["hrv"] = sat_hrv_datapipe

    if use_pv:
        logger.debug("Opening PV")
        pv_datapipe = OpenPVFromNetCDF(
            pv=conf_in.pv
        ).add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=conf_in.pv.history_minutes),
        )

        used_datapipes["pv"] = pv_datapipe

    if use_topo:
        logger.debug("Opening Topographic Data")
        topo_datapipe = OpenTopography(conf_in.topographic.topographic_filename)

        used_datapipes["topo"] = topo_datapipe

    return used_datapipes


def get_and_return_overlapping_time_periods_and_t0(datapipes_dict: dict, key_for_t0: str = "gsp"):
    """
    Takes datapipes and obtains the overlapping time periods + t0 time datapipes

    Args:
        datapipes_dict: Dictionary of datapipes to compute the time intersection of
        key_for_t0: Key to use for the t0 datapipe

    Returns:
        Dictionary of datapipes with the proper time slices selected
    """
    assert key_for_t0 in datapipes_dict
        
    time_period_datapipes = []  # Using later to compute intersections
    datapipes_to_return = {}  # Returned along with original ones
    configuration = datapipes_dict.pop("config")
    
    datapipes_dict[key_for_t0], t0_datapipe = datapipes_dict[key_for_t0].fork(2, buffer_size=5)
    
    for key in datapipes_dict.keys():
        if "topo" in key:
            continue

        elif key == "nwp":
            sample_frequency = 180 # Init times are 3 hours apart
            history_duration = configuration.input_data.nwp.history_minutes
            forecast_duration = configuration.input_data.nwp.forecast_minutes
            time_dim="init_time_utc"

        elif key ==  "sat":            
            sample_frequency = 5
            history_duration = configuration.input_data.satellite.history_minutes
            forecast_duration = 0
            time_dim="time_utc"

        elif key == "hrv":
            sample_frequency = 5
            history_duration = configuration.input_data.hrvsatellite.history_minutes
            forecast_duration = 0
            time_dim="time_utc"

        elif key == "pv":
            sample_frequency = 5
            history_duration = configuration.input_data.pv.history_minutes
            forecast_duration = configuration.input_data.pv.forecast_minutes
            time_dim="time_utc"
            
        elif key == "gsp":
            sample_frequency = 30
            history_duration = configuration.input_data.gsp.history_minutes
            forecast_duration = configuration.input_data.gsp.forecast_minutes
            time_dim="time_utc"
        
        else:
            raise ValueError(f"Unexpected key: {key}")
            
        datapipes_to_return[key], time_periods = datapipes_dict[key].fork(2, buffer_size=5)
            
        time_periods = time_periods.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=sample_frequency),  
            history_duration=timedelta(minutes=history_duration),
            forecast_duration=timedelta(minutes=forecast_duration),
            time_dim=time_dim,
        )
        
        time_period_datapipes.append(time_periods)

    # Now have the forked ones
    # find joint overlapping timer periods
    if len(time_period_datapipes)>1:
        logger.debug("Getting joint time periods")
        overlapping_datapipe = time_period_datapipes[0].select_overlapping_time_slice(
            secondary_datapipes=time_period_datapipes[1:],
        )
    else:
        logger.debug("Skipping getting joint time periods")
        overlapping_datapipe = time_period_datapipes[0]
        
    # select time periods
    t0_datapipe = (
        t0_datapipe.select_time_periods(time_periods=overlapping_datapipe)
        .select_t0_time(return_all_times=False)
    )

    num_t0_datapipes = len(datapipes_to_return)  # One for each input
    
    if num_t0_datapipes>1:
        t0_datapipes = t0_datapipe.fork(num_t0_datapipes, buffer_size=5)
    else:
        t0_datapipes = [t0_datapipe]
    
    for key, datapipe in zip(list(datapipes_to_return.keys()), t0_datapipes):
        datapipes_to_return[key + "_t0"] = datapipe
    
    # Reattach ignored entries
    datapipes_to_return["config"] = configuration
    if "topo" in datapipes_dict:
        datapipes_to_return["topo"] = datapipes_dict["topo"]
    
    return datapipes_to_return


def add_selected_time_slices_from_datapipes(used_datapipes: dict, split_future=True):
    """
    Takes datapipes and t0 datapipes and returns the sliced datapipes

    Args:
        used_datapipes: Dictionary of used datapipes and t0 ones
        split_future: Boolean value of whether to split the history and
            future of '"pv"` and `"gsp"` into `"key"` and `"key_future"`
           entries in the datapipe dict

    Returns:
        Dictionary of datapipes after the time slices are selected
    """
    datapipes_to_return = {}  # Returned along with original ones
    configuration = used_datapipes.pop("config")
        
    for key in ["gsp", "pv"]:
        if key in used_datapipes.keys() and split_future:
            used_datapipes[f"{key}_future"], used_datapipes[key] = used_datapipes[key].fork(2, buffer_size=5)
            used_datapipes[f"{key}_future_t0"], used_datapipes[f"{key}_t0"] = used_datapipes[f"{key}_t0"].fork(2, buffer_size=5)
    
    for key, datapipe in used_datapipes.items():
        if ("topo" in key) or ("_t0" in key):
            continue
           
        elif key == "nwp":
            datapipe_method = datapipe.convert_to_nwp_target_time
            sample_frequency_mins = 60
            history_mins = configuration.input_data.nwp.history_minutes
            forecast_mins = configuration.input_data.nwp.forecast_minutes
            
        elif key == "sat":
            datapipe_method = datapipe.select_time_slice
            sample_frequency_mins = 5
            history_mins = configuration.input_data.satellite.history_minutes
            forecast_mins = 0

        elif key == "hrv":
            datapipe_method = datapipe.select_time_slice
            sample_frequency_mins = 5
            history_mins = configuration.input_data.hrvsatellite.history_minutes
            forecast_mins = 0
            
        elif key == "pv":
            datapipe_method = datapipe.select_time_slice
            sample_frequency_mins = 5
            history_mins = configuration.input_data.pv.history_minutes
            forecast_mins = 0 if split_future else configuration.input_data.pv.forecast_minutes
            
        elif key == "pv_future":
            datapipe_method = datapipe.select_time_slice
            sample_frequency_mins = 5
            history_mins = 0
            forecast_mins = configuration.input_data.pv.forecast_minutes
            
        elif key == "gsp":
            datapipe_method = datapipe.select_time_slice
            sample_frequency_mins = 30
            history_mins = configuration.input_data.gsp.history_minutes
            forecast_mins = 0 if split_future else configuration.input_data.gsp.forecast_minutes
            
        elif key == "gsp_future":
            datapipe_method = datapipe.select_time_slice
            sample_frequency_mins = 30
            history_mins = 0
            forecast_mins = configuration.input_data.gsp.forecast_minutes
            
        else:
            raise ValueError(f"Uneexpected key: {key}")
        
        datapipes_to_return[key] = datapipe_method(
            t0_datapipe=used_datapipes[key + "_t0"],
            sample_period_duration=timedelta(minutes=sample_frequency_mins),
            history_duration=timedelta(minutes=history_mins),
            forecast_duration=timedelta(minutes=forecast_mins),
        )

    if "topo" in used_datapipes:
        datapipes_to_return["topo"] = used_datapipes["topo"]
    datapipes_to_return["config"] = configuration
    
    return datapipes_to_return
