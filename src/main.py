import scipy.stats as ss
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf 
import datetime
from formulas import * 
import pandas as pd 

ticker = yf.Ticker("AAPL")
x = ticker.history(period="1d")['Close'].iloc[-1] 
expiration = ticker.options[0]
print(expiration)
option_chain = ticker.option_chain(expiration)
calls = option_chain.calls

today = datetime.date.today()
expiry = datetime.date.fromisoformat(expiration)
T_days = (expiry - today).days
tt = T_days / 252

K = calls['strike']
market_prices = calls['lastPrice']

## Random volatility 
bs = [black_scholes_model(x, 0, c, 0.05, 0.05, tt) for c in K]

## Historical Volatility 
data = ticker.history(period="3mo")['Close']
his_vol = historical_volatility(data,30).iloc[-1]
print("hist_vol:", his_vol)

bs_hist_vol = [black_scholes_model(x, 0, c, 0.05, his_vol, tt) for c in K]

## Bootstrap Historical Volatiltity 
r = len(data)
boot_vol_dist = bootstrap_volatility(data,r, 5000)

lower = np.quantile(boot_vol_dist, 0.025)
upper = np.quantile(boot_vol_dist, 0.975)

print(lower, upper)

plt.hist(boot_vol_dist)
boot_vol = boot_vol_dist.mean()
print('Boot vol:', boot_vol)

bs_boot_vol = [black_scholes_model(x, 0, c, 0.05, boot_vol, tt) for c in K]

## Implied Volatility 
K, market_prices = zip(*[(k, mp) for k, mp in zip(K, market_prices) if mp > 0 and np.isfinite(mp)])
implied_vols = []
for k, mp in zip(K, market_prices):
    iv = implied_volatility(mp, x, 0, k, 0.05, 0.2, tt)
    implied_vols.append(iv)
   

iv_nona = [iv for iv in implied_vols if np.isfinite(iv)]
chose_iv = np.median(iv_nona)
print('Implied vol:' , chose_iv)

bs_implied_vol = [black_scholes_model(x, 0, c, 0.05, chose_iv, tt) for c in K]

## Random Volatility 
plt.figure(figsize=(10,6))
plt.plot(K, market_prices, label="Market Price", marker='o')
plt.plot(K, bs, label="Black-Scholes Price", marker='x')
plt.axvline(x, color='gray', linestyle='--', label='Spot Price')
plt.title(f"AAPL Call Option Prices (Exp: {expiration})")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.show()
## Historical Volatility 
plt.figure(figsize=(10,6))
plt.plot(K, market_prices, label="Market Price", marker='o')
plt.plot(K, bs_hist_vol, label="Black-Scholes Price", marker='x')
plt.axvline(x, color='gray', linestyle='--', label='Spot Price')
plt.title(f"AAPL Call Option Prices (Exp: {expiration}) with historical volatility")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.show()

## Bootstrap Volatility 
plt.figure(figsize=(10,6))
plt.plot(K, market_prices, label="Market Price", marker='o')
plt.plot(K, bs_boot_vol, label="Black-Scholes Price", marker='x')
plt.axvline(x, color='gray', linestyle='--', label='Spot Price')
plt.title(f"AAPL Call Option Prices (Exp: {expiration}) with bootstrap historical volatility")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.show()

## Implied Volatility 
plt.figure(figsize=(10,6))
plt.plot(K, market_prices, label="Market Price", marker='o')
plt.plot(K, bs_implied_vol, label="Black-Scholes Price", marker='x')
plt.axvline(x, color='gray', linestyle='--', label='Spot Price')
plt.title(f"AAPL Call Option Prices (Exp: {expiration}) with implied volatility")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.show()

rmse_random= RMSE(np.array(market_prices), np.array(bs))
print('error random:' , rmse_random)

rmse_hist = RMSE(np.array(market_prices), np.array(bs_hist_vol))
print('error hist:', rmse_hist)

rmse_iv = RMSE(np.array(market_prices), np.array(bs_implied_vol))
print('error implied:' , rmse_iv)

rmse_boot = RMSE(np.array(market_prices), np.array(bs_boot_vol))
print('error boot:', rmse_boot)

market_prices = np.array(market_prices)

relative_rmse = rmse_random/np.mean(market_prices)
print(relative_rmse)

rmse_random - rmse_hist