""" Utils Functions to for fake data """

from datetime import timedelta

import numpy as np
import pandas as pd


def make_t0_datetimes_utc(batch_size, temporally_align_examples: bool = False):
    """
    Make list of t0 datetimes

    Args:
        batch_size: the batch size
        temporally_align_examples: option to align examples (within the batch) in time

    Returns: pandas index of t0 datetimes
    """

    all_datetimes = pd.date_range("2023-01-01", "2023-02-01", freq="5min")

    if temporally_align_examples:
        t0_datetimes_utc = list(np.random.choice(all_datetimes, size=1)) * batch_size
    else:
        if len(all_datetimes) >= batch_size:
            replace = False
        else:
            # there are not enought data points,
            # so some examples will have the same datetime
            replace = True

        t0_datetimes_utc = np.random.choice(all_datetimes, batch_size, replace=replace)
    # np.random.choice turns the pd.Timestamp objects into datetime.datetime objects.

    t0_datetimes_utc = pd.to_datetime(t0_datetimes_utc)

    # TODO make test repeatable using numpy generator
    # https://github.com/openclimatefix/nowcasting_dataset/issues/594

    return t0_datetimes_utc


def get_n_time_steps_from_config(input_data_configuration, include_forecast: bool = True) -> int:
    """
    Get the number of time steps from the input data configuration

    Args:
        input_data_configuration: NWP, GSP, PV, e.t.c see ocf_datapipes.config.model
        include_forecast: option to include forecast timesteps or not

    Returns: number of time steps for this input data

    """
    # get history time steps
    n_time_steps = int(
        input_data_configuration.history_minutes / input_data_configuration.time_resolution_minutes
    )

    if include_forecast:
        n_time_steps = n_time_steps + int(
            input_data_configuration.forecast_minutes
            / input_data_configuration.time_resolution_minutes
        )

    # add extra step for now
    n_time_steps = n_time_steps + 1

    return n_time_steps


def make_time_utc(
    batch_size, history_minutes, forecast_minutes, t0_datetime_utc, time_resolution_minutes
):
    """
    Make time utc array

    Args:
        batch_size: the batch size
        history_minutes: the history minutes we want
        forecast_minutes: the amount of minutes that the forecast will be fore
        t0_datetime_utc: t0_datetime
        time_resolution_minutes: time resolution e.g 5 for daat in 5 minutes chunks

    Returns: array of time_utc

    """
    start_datetime = t0_datetime_utc - timedelta(minutes=history_minutes)
    end_datetime = t0_datetime_utc + timedelta(minutes=forecast_minutes)
    time_utc = pd.date_range(
        start=start_datetime, end=end_datetime, freq=f"{time_resolution_minutes}min"
    )

    time_utc = time_utc.values.astype("datetime64[s]")
    time_utc = np.tile(time_utc, (batch_size, 1))
    return time_utc
