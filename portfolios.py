'''
Functions for wrangling CRSP data. 

Quantile portfolios. Portfolio returns. Summary table.
'''

import utils

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

'''
Lagging returns
'''

def lag_return_permno(data,n,permno):
    # Helper for lag_returns
    permno_data = data.loc[data['PERMNO'] == permno]
    for lag in range(1,n+1):
        col = 'LRET_{}'.format(lag)
        permno_data[col] = permno_data['RET'].shift(lag)
    return permno_data.iloc[3:]

def lag_returns(data,n):
    # Adds lagged returns (up to n lags) of each stock to the dataset.
    # Loses n first observations for each stock.
    permnos = data['PERMNO'].unique()
    dfs = threadify(lambda permno: lag_return_permno(data,n,permno),permnos)
    return pd.concat(dfs, ignore_index=True)

'''
Quantile tables
'''

def quantiles_for_date(data,wrt,date,n):
    # Helper for quantile_table
    r = 100//n
    data = data.loc[data['date'] == date]
    qs = [q/100 for q in range(r,100+r,r)]
    breakpoints = [np.quantile(data[wrt].values,q) for q in qs]
    return {**{'date':date}, **dict(zip(range(1,r+1),breakpoints))}

def quantile_table(data,wrt,n):
    # Returns DataFrame of quantile breakpoints with respect to variable (wrt).
    # n is the number breakpoints formed, e.g. with n = 10, 10 breakpoints are formed
    # which can used to assign stock into 10 portfolios with respect to the variable.
    dates = data['date'].unique()
    qs_dates = threadify(lambda date: quantiles_for_date(data,wrt,date,n),dates)
    return pd.DataFrame(qs_dates)

'''
Forming portfolios
'''

from bisect import bisect

def assign_portfolio(breakpoints,date,var):
    # Helper for portfolios_for_date (form_portfolios)
    s = len(breakpoints.columns) - 1
    bp = breakpoints.loc[breakpoints['date'] == date][range(1,s+1)].values.flatten()
    pos = bisect(bp,var) + 1
    if pos > s: pos = s
    return pos

from collections import OrderedDict

def portfolios_for_date(data,breakpoints,date,wrt):
    # Helper for form_portfolios
    df = data.loc[data['date'] == date]
    name = 'PORT_' + str(wrt)
    df[name] = df['ME'].map(lambda e: assign_portfolio(breakpoints,date,e))
    print("{} ".format(date), end='')
    return df

import time

def form_portfolios(data,wrt,n):
    # Returns DataFrame (in CRSP form). Each stock assigned into quantile portfolio with
    # respect to variable (wrt). Total n quantile portfolios.
    start = time.time()
    print('Progress...')
    breakpoints = quantile_table(data,wrt,n)
    dates = data['date'].unique()
    dfs = threadify(lambda d: portfolios_for_date(data,breakpoints,d,wrt),dates)
    print('Done. Execution time: {}s'.format(round(time.time()-start,3)))
    return pd.concat(dfs, ignore_index=True)

'''
Summary tables
'''

def portfolios_summary_month(data,date,wrt):
    # Helper for portfolios_summary_table
    name = 'PORT_' + str(wrt)
    return {**{'date':date}, **data.loc[data['date'] == date][name].value_counts().to_dict()}

def portfolios_summary_table(data,wrt):
    # Returns DataFrame. For each date the number of stocks in the quantile portfolios.
    dates = data['date'].unique()
    df = pd.DataFrame(threadify(lambda d: portfolios_summary_month(data,d,wrt),dates)).set_index('date')
    return df.sort_index(axis=1)


'''
Returns
'''

def portfolios_returns_mean(data,date,wrt,n):
    # Helper for portfolios_returns_mean_table
    returns = {}
    portfolios = range(1,n+1)
    for i in portfolios:
        name = 'PORT_' + str(wrt)
        p_data = data.loc[(data['date'] == date) & (data[name] == i)]
        returns[i] = p_data["RET"].values.mean()
    return {**{'date':date}, **returns}

def portfolios_returns_mean_table(data,wrt,n):
    # Returns DataFrame. Time series returns of the quantile portfolios.
    dates = data['date'].unique()
    returns = threadify(lambda e: portfolios_returns_mean(data,e,wrt,n),dates)
    return pd.DataFrame(returns).set_index('date')