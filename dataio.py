'''
Importing and cleaning data
'''

from utils import threadify

import pandas as pd
import numpy as np

'''
Table of returns
'''

def permno_to_timeseries(data,permno):
    # Helper for crsp_to_timeseries
    df = data.loc[data['PERMNO'] == permno][['date','RET']].set_index('date')
    df.columns = [permno]
    return df

def ret_table(data):
    # Returns timeseries DataFrame of unique permnos
    permnos = data['PERMNO'].unique()
    dfs = threadify(lambda e: permno_to_timeseries(data,e),permnos)
    dfs = threadify(lambda df: df.loc[~df.index.duplicated()],dfs)
    table =  pd.concat(dfs,axis=1,sort=False)
    table.index = table.index.values.astype(int)
    return table.fillna(0)

'''
Table of volatility
'''

def vol_table(ret_table,n):
    table = ret_table.rolling(n).std().fillna(method='bfill').fillna(method='ffill')
    table.index = table.index.values.astype(int)
    return table