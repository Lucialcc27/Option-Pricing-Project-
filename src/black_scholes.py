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


# Get AAPL option data
ticker = yf.Ticker("AAPL")
x = ticker.history(period="1d")['Close'].iloc[-1] # Spot price
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