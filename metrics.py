'''
Summary statistics for time series returns
'''

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

"""
Summary table of basics
"""

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

def monthly_turnover(positions):
    pass

def summary_statistics(returns):
    rows = [mean_returns(returns),
            total_returns(returns),
            max_return(returns),
            min_return(returns),
            realized_vola(returns),
            max_drawdown(returns)]
    return pd.DataFrame(rows).set_index('Index')

"""
Summary table of timeseries regressions (asset pricing models e.g. CAPM, FF3, FF5)
"""

import collections

def signif_stars(pvalue): 
    if pvalue <= 0.01: return '***'
    elif pvalue <= 0.05: return '**'
    elif pvalue <= 0.10: return '*'
    else: return ''

def alphas_tstats_portfolio(portfolio,data):
    capm = ols(portfolio + '~ MKT', data).fit()
    ff_3 = ols(portfolio + '~ MKT + SMB + HML', data).fit()
    ff_5 = ols(portfolio + '~ MKT + SMB + HML + RMW + CMA', data).fit()
    alphas = {'CAPM alpha' : capm.params['Intercept'], 'FF3 alpha' : ff_3.params['Intercept'], 'FF5 alpha' : ff_5.params['Intercept']}
    tstats = {'CAPM tstat' : capm.tvalues['Intercept'], 'FF3 tstat' : ff_3.tvalues['Intercept'], 'FF5 tstat' : ff_5.tvalues['Intercept']}
    pvalues = {'CAPM*' : signif_stars(capm.pvalues['Intercept']), 
               'FF3*' : signif_stars(ff_3.pvalues['Intercept']), 
               'FF5*' : signif_stars(ff_5.pvalues['Intercept'])}
    d = {**{'Portfolio' : portfolio}, **alphas, **tstats, **pvalues}
    return collections.OrderedDict(sorted(d.items()))

def time_series_regressions(portfolios,data):
    rows = []
    for p in portfolios:
        rows.append(alphas_tstats_portfolio(p,data))
    return pd.DataFrame(rows).set_index('Portfolio')
