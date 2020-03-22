'''
Importing and cleaning data
'''

from utils import threadify

import pandas as pd
import numpy as np

def permno_to_timeseries(data,permno):
    # Helper for crsp_to_timeseries
    df = data.loc[data['PERMNO'] == permno][['date','RET']].set_index('date')
    df.columns = [permno]
    return df

def crsp_to_timeseries(data):
    # Returns timeseries DataFrame of unique permnos
    permnos = data['PERMNO'].unique()
    dfs = threadify(lambda e: permno_to_timeseries(data,e),permnos)
    dfs = threadify(lambda df: df.loc[~df.index.duplicated()],dfs)
    return pd.concat(dfs,axis=1,sort=False)