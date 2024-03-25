""" Data pipes and utils for getting PV data from database"""

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from sqlalchemy import text
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.load.pv.utils import put_pv_data_into_an_xr_dataarray

logger = logging.getLogger(__name__)


@functional_datapipe("open_pv_from_pvsites_db")
class OpenPVFromPVSitesDBIterDataPipe(IterDataPipe):
    """Data pipes and utils for getting PV data from pvsites database"""

    def __init__(
        self,
        history_minutes: int = 30,
    ):
        """
        Datapipe to get PV from pvsites database

        Args:
            history_minutes: How many history minutes to use
        """

        super().__init__()

        self.history_minutes = history_minutes
        self.history_duration = pd.Timedelta(self.history_minutes, unit="minutes")

    def __iter__(self):
        df_metadata = get_metadata_from_pvsites_database()
        df_gen = get_pv_power_from_pvsites_database(history_duration=self.history_duration)

        # Database record is very short. Set observed max to NaN
        df_metadata["observed_capacity_wp"] = np.nan

        # Ensure systems are consistant between generation data, and metadata
        common_systems = list(np.intersect1d(df_metadata.index, df_gen.columns))
        df_gen = df_gen[common_systems]
        df_metadata = df_metadata.loc[common_systems]

        # Compile data into an xarray DataArray
        xr_array = put_pv_data_into_an_xr_dataarray(
            df_gen=df_gen,
            observed_system_capacities=df_metadata.observed_capacity_wp,
            nominal_system_capacities=df_metadata.nominal_capacity_wp,
            ml_id=df_metadata.ml_id,
            latitude=df_metadata.latitude,
            longitude=df_metadata.longitude,
            tilt=df_metadata.get("tilt"),
            orientation=df_metadata.get("orientation"),
        )

        logger.info(f"Found {len(xr_array.ml_id)} PV systems")

        while True:
            yield xr_array


def get_metadata_from_pvsites_database() -> pd.DataFrame:
    """Load metadata from the pvsites database"""
    # make database connection
    url = os.getenv("DB_URL_PV")
    db_connection = DatabaseConnection(url=url, base=Base_Forecast)

    with db_connection.engine.connect() as conn:
        df_sites_metadata = pd.DataFrame(conn.execute(text("SELECT * FROM sites")).fetchall())

    df_sites_metadata["nominal_capacity_wp"] = df_sites_metadata["capacity_kw"] * 1000

    df_sites_metadata = df_sites_metadata.set_index("site_uuid")

    return df_sites_metadata


def get_pv_power_from_pvsites_database(history_duration: timedelta):
    """Load recent generation data from the pvsites database"""

    # make database connection
    url = os.getenv("DB_URL_PV")
    db_connection = DatabaseConnection(url=url, base=Base_Forecast)

    columns = "site_uuid, generation_power_kw, start_utc, end_utc"

    start_time = f"{datetime.now() - history_duration}"

    with db_connection.engine.connect() as conn:
        df_db_raw = pd.DataFrame(
            conn.execute(
                text(f"SELECT {columns} FROM generation where end_utc >= '{start_time}'")
            ).fetchall()
        )

    # Reshape
    df_gen = df_db_raw.pivot(index="end_utc", columns="site_uuid", values="generation_power_kw")

    # Rescale from kW to W
    df_gen = df_gen * 1000

    # Fix data types
    df_gen = df_gen.astype(np.float32)

    return df_gen
