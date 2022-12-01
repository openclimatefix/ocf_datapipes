"""Select PV systems and Dates with No output for the entire day"""
import logging

import xarray as xr
import pandas as pd
import numpy as np
from typing import Union, List, Dict

from ocf_datapipes.load.pv.utils import xr_to_df
from ocf_datapipes.load.pv.utils import dates_list as dt_list

    
class DatesOfPVSystemsWithoutOutput:
    """Returns a dataframe with one column of pv system ids and another column
    
    of dates of those ids with no pv output """
    def __init__(
        self,
        pv_power:xr.Dataset)->None:
        """

        Args: 
            pv_power: pv.netcdf file which has pv system ids as variables
                     and dates as coordniates, and their corresponding pv output
        """

        self.pv_power = pv_power
 
    def __iter__(
        self
        )-> pd.DataFrame():
        """

        This function gives a pandas dataframe that stores 
        all the SSID with corresponding dates with no PV output
        
        """
        no_pv_df = pd.DataFrame()
        ssid_list = list(self.pv_power)
        dates_list = dt_list(pv_power = self.pv_power)
        for i in ssid_list:
            for j in dates_list:
                # xr_to_df function gives a dataframe for one ssid 
                # and its corresponsing date
                df = xr_to_df(
                    pv_power_xr = self.pv_power,
                    ssid=i,
                    date_oi=j)
                df_values = df.values
                torf = np.isnan(df_values).all()
                if torf == False:
                    continue
                temp = pd.DataFrame(
                    {
                        'ssid': i,
                        'date':j
                    }, index = [0]
                )
                no_pv_df = pd.concat([no_pv_df, temp])
            print("checking for no power in a day for PV system with ssid",i,"has been completed")
            print(len(self.ssid_list) - self.ssid_list.index(i),"systems are left")
        return no_pv_df