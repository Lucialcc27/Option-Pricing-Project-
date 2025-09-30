import scipy.stats as ss
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf 
import datetime


# Black-Scholes formula

def black_scholes_model(x, t, c, r, v, tt):
    if tt <= t or v <= 0 or c <= 0:
        return np.nan
    d_1 = (np.log(x/c) + (r + 0.5*v**2)*(tt-t)) / (v * np.sqrt(tt-t))
    d_2 = d_1 - v * np.sqrt(tt-t)
    w = x * ss.norm.cdf(d_1) - c * np.exp(-r*(tt-t)) * ss.norm.cdf(d_2)
    return w



def historical_volatility(history_data,n):
    returns = history_data.pct_change()
    hist_vol = returns.rolling(window=n).std()*np.sqrt(252)
    return hist_vol

def vega(x, t, c, r, v, tt):
     d_1 = (np.log(x/c) + (r + 0.5*v**2)*(tt-t)) / (v * np.sqrt(tt-t))
     vega = x * ss.norm.pdf(d_1) * np.sqrt(tt-t)
     return vega

def implied_volatility(b, x, t, c, r, v_0, tt, tol=1e-5, max_iteration=100):
    sigma = v_0
    for i in range(max_iteration): 
        bs_price = black_scholes_model(x, t, c, r, sigma, tt)
        if not np.isfinite(bs_price):
            return np.nan
        
        v = vega(x, t, c, r, sigma, tt)
        if v < 1e-8:
            return np.nan
        
        diff = bs_price - b
        if abs(diff) < tol: 
            return sigma
        
        sigma -= diff / v

    return np.nan