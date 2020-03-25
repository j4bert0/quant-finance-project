'''
Portfolio position sizing
'''

from utils import threadify
from scipy.optimize import minimize
import numpy as np

def optimized_weights_for_covm(covm, lowerbound=0):
    t = lambda X: np.transpose(X)
    portfolio_var = lambda w: np.matmul(np.matmul(t(w),covm),w)
    n = covm.shape[0]
    w = np.repeat(1, n)/n
    
    # If estimation of covariance matrix fails, returns equal weights
    if np.isnan(covm).any(): return w
    
    # Weights sum to 1
    cons = [{'type': 'eq', 'fun': lambda w:  np.sum(w)-1.0}]
    # No short sales
    b = [(lowerbound, None) for i in range(n)]
    
    res = minimize(portfolio_var, w, method='SLSQP',constraints=cons, bounds=b)
    
    # Returns minimization successful return optimal weights
    # Else return equal weights
    if res.success:
        return res.x
    else:
        return w

def cov_matrix_portfolio(permnos,ret_table,date):
    return ret_table.loc[:date][permnos].cov().to_numpy()

def optimized(permnos,ret_table,lowerbound,date):
    covm = cov_matrix_portfolio(permnos,ret_table,date)
    return optimized_weights_for_covm(covm, lowerbound=0)

def vol_scaled_weights(permnos,vol_table,date):
    vols = vol_table[permnos].loc[date].values
    pos = vols/sum(vols)
    if np.isnan(pos).any(): 
        return equal_weights(permnos,date)
    return pos

def inv_vol_scaled_weights(permnos,vol_table,date):
    vols = 1 / vol_table[permnos].loc[date].values
    pos = vols/sum(vols)
    if np.isnan(pos).any(): 
        return equal_weights(permnos,date)
    return pos

def equal_weights(permnos,date):
    n = len(permnos)
    return np.repeat(1,n)/n

def rand_weights_weights(permnos):
    pass