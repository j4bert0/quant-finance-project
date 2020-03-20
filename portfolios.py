'''
Functions for wrangling CRSP data. 

Quantile portfolios. Portfolio returns. Summary table. Plots.
'''

from utils import threadify

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


'''
Filter universe
'''

def filter_me_date(data,date,n):
    # Helper for filter_me
    return data.loc[(data['date'] == date) & (data['PORT_ME'] > n)]

def filter_me(data,n):
    # Removes n smallest deciles (10%) by market equity from data
    dates = data['date'].unique()
    assigned_data = form_portfolios(data,'ME',10)
    dfs = threadify(lambda d: filter_me_date(assigned_data,d,n),dates)
    return pd.concat(dfs, ignore_index=True).drop('PORT_ME',1)

'''
Lagging returns
'''

def lag_return_permno(data,n,permno):
    # Helper for lag_returns
    permno_data = data.loc[data['PERMNO'] == permno]
    for lag in range(1,n+1):
        col = 'LRET_{}'.format(lag)
        permno_data[col] = permno_data['RET'].shift(lag)
    return permno_data.iloc[n:-n]

def lag_returns(data,n):
    # Adds lagged returns (up to n lags) of each stock to the dataset.
    # Loses n first and last observations for each stock.
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
    return {**{'date':date}, **dict(zip(range(1,n+1),breakpoints))}

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
    df[name] = df[wrt].map(lambda e: assign_portfolio(breakpoints,date,e))
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

"""
Portfolio for date (for debugging)
"""

def portfolio_at_date(assigned_data,date,wrt,n):
    # Assumes assigned data (data with portfolios 'PORT_...' variable).
    # Returns table of stocks of the portfolio.
    name = 'PORT_' + str(wrt)
    portfolio = assigned_data.loc[(assigned_data['date'] == date) & (assigned_data[name] == n)]
    return portfolio[['TICKER','COMNAM',wrt,'RET']]

'''
Summary tables
'''

def portfolios_size_summary_month(data,date,wrt):
    # Helper for portfolios_size_summary
    name = 'PORT_' + str(wrt)
    return {**{'date':date}, **data.loc[data['date'] == date][name].value_counts().to_dict()}

def portfolios_size_summary(data,wrt):
    # Returns DataFrame. For each date the number of stocks in the quantile portfolios.
    dates = data['date'].unique()
    df = pd.DataFrame(threadify(lambda d: portfolios_size_summary_month(data,d,wrt),dates)).set_index('date')
    return df.sort_index(axis=1)

def portfolios_me_summary_month(data,date,wrt,n):
    # Helper for portfolios_me_summary
    name = 'PORT_' + str(wrt)
    d = {'date':date}
    for i in range(1,n+1):
        d[i] = data.loc[(data['date'] == date) & (data[name] == i)]['ME'].values.mean()
    return d

def portfolios_me_summary(data,wrt,n):
    # Returns DataFrame. For each date the average market equity in the quantile portfolios.
    dates = data['date'].unique()
    df = pd.DataFrame(threadify(lambda d: portfolios_me_summary_month(data,d,wrt,n),dates)).set_index('date')
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

'''
Plots
'''

def plot_cumulative(returns,log_scale=False,portfolios=[]):
    # Plots DataFrame of portfolios returns.
    s = len(portfolios)
    if s == 0:
        c_returns = returns.copy()
        s = len(returns.columns)
    else:
        c_returns = returns
    c_returns.loc['0'] = [0] * s
    df = (1 + c_returns.sort_index()).cumprod()
    return df.plot(title='Cumulative returns of portfolios',logy=log_scale,figsize=(10,6))

def plot_returns(returns):
    return returns.plot(title='Returns of portfolios',figsize=(10,6))

def plot_boxplot(returns):
    return returns.boxplot(title='Boxplot of portfolios',figsize=(10,6))

    