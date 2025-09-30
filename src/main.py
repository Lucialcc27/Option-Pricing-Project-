import scipy.stats as ss
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf 
import datetime
from formulas import *

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

bs = [black_scholes_model(x, 0, c, 0.05, 0.05, tt) for c in K]

data = ticker.history(period="3mo")['Close']
his_vol = historical_volatility(data,30).iloc[-1]
print(his_vol)

bs_hist_vol = [black_scholes_model(x, 0, c, 0.05, his_vol, tt) for c in K]

K, market_prices = zip(*[(k, mp) for k, mp in zip(K, market_prices) if mp > 0 and np.isfinite(mp)])
implied_vols = []
for k, mp in zip(K, market_prices):
    iv = implied_volatility(mp, x, 0, k, 0.05, 0.2, tt)
    implied_vols.append(iv)
   

iv_nona = [iv for iv in implied_vols if np.isfinite(iv)]
chose_iv = np.median(iv_nona)
print(chose_iv)

bs_implied_vol = [black_scholes_model(x, 0, c, 0.05, chose_iv, tt) for c in K]

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

errors = abs(np.array(bs) - np.array(market_prices))
plt.figure(figsize=(10,6))
plt.plot(K, errors, label="Model Error (BS - Market)")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Strike Price")
plt.ylabel("Error ($)")
plt.title("Black-Scholes Pricing Error")
plt.grid(True)
plt.legend()
plt.show()

error_hist =abs(np.array(bs_hist_vol) - np.array(market_prices))
plt.figure(figsize=(10,6))
plt.plot(K, error_hist, label="Model Error (BS - Market) historical volatility")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Strike Price")
plt.ylabel("Error ($)")
plt.title("Black-Scholes Pricing Error")
plt.grid(True)
plt.legend()
plt.show()

error_iv = abs(np.array(bs_implied_vol) - np.array(market_prices))
plt.figure(figsize=(10,6))
plt.plot(K, error_iv, label="Model Error (BS - Market) implied volatility")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Strike Price")
plt.ylabel("Error ($)")
plt.title("Black-Scholes Pricing Error implied volatility")
plt.grid(True)
plt.legend()
plt.show()
