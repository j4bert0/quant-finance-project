'''
Summary statistics for time series returns
'''

import pandas as pd
import numpy as np

def mean_returns(returns):
    d = dict(round(returns.mean()*100,2))
    d['Index'] = 'Mean return (month)'
    return d

def total_returns(returns):
    ret = (1 + returns).cumprod().iloc[-1] - 1
    d = dict(round(ret*100,2))
    d['Index'] = 'Total return %'
    return d

def max_return(returns):
    d = dict(round(returns.max()*100,2))
    d['Index'] = 'Best month'
    return d

def min_return(returns):
    d = dict(round(returns.min()*100,2))
    d['Index'] = 'Worst month'
    return d

def realized_vola(returns):
    d = dict(round(returns.std()*100,2))
    d['Index'] = 'Realized volatility (month)'
    return d

def sharpe_ratios(returns):
    pass

def sortino_ratios(returns):
    pass

def max_drawdown(returns):
    drawdowns = returns.cummax()-returns.cummin()
    d = dict(round(drawdowns.max()*100,2))
    d['Index'] = 'Max drawdown'
    return d

def summary_statistics(returns):
    rows = [mean_returns(returns),
            total_returns(returns),
            max_return(returns),
            min_return(returns),
            realized_vola(returns),
            max_drawdown(returns)]
    return pd.DataFrame(rows).set_index('Index')

def time_series_regressions(returns,factors):
    pass