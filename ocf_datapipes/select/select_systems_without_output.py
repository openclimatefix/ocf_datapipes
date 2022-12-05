"""Select PV systems and Dates with No output for the entire day"""
import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.load.pv.utils import dates_list as dt_list
from ocf_datapipes.load.pv.utils import xr_to_df

logger = logging.getLogger(__name__)


@functional_datapipe("select_pv_id_dates_with_no_pv_output")
class DatesOfPVSystemsWithoutOutputIterDataPipe(IterDataPipe):
    """
    Returns a dataframe with one column of pv system ids and

    another column of dates of those ids with no pv output
    """

    def __init__(self, source_datapipe: IterDataPipe) -> None:
        """This function gives a pandas dataframe that stores all the SSID

        with corresponding dates with no PV output

        Args:
            source_datapipe: After loading pv.netcdf file into Xarray Dataset
                        which has pv system ids as variables
                        and dates as coordniates, and their corresponding pv output
        """

        self.source_datapipe = source_datapipe

    def __iter__(self) -> pd.DataFrame():

        for xr_dataset in self.source_datapipe:
            no_pv_df = pd.DataFrame()
            ssid_list = list(xr_dataset)
            dates_list = dt_list(pv_power=xr_dataset)
            for i in ssid_list:
                for j in dates_list:
                    # xr_to_df function gives a dataframe for one ssid
                    # and its corresponsing date
                    df = xr_to_df(pv_power_xr=xr_dataset, ssid=i, date_oi=j)
                    df_values = df.values
                    torf = np.isnan(df_values).all()
                    if torf == False:
                        continue
                    temp = pd.DataFrame({"ssid": i, "date": j}, index=[0])
                    no_pv_df = pd.concat([no_pv_df, temp])
                print(
                    "checking for no power in a day for PV system with ssid",
                    i,
                    "has been completed",
                )
                print(len(ssid_list) - ssid_list.index(i), "systems are left")
                yield no_pv_df
