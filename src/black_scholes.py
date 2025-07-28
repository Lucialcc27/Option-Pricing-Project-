import scipy.stats as ss
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf 


# Black-Scholes formula

def black_scholes_model(x, t, c, r, v, tt): 
    d_1 = (np.log(x/c) + (r + 0.5*v**2)*(tt-t)) / (v * np.sqrt(tt-t))
    d_2 = d_1 - v * np.sqrt(tt-t)
    w = x * ss.norm.cdf(d_1) - c * np.exp(-r*(tt-t)) * ss.norm.cdf(d_2)
    return w 

print(black_scholes_model(100, 0, 80, 0.01,0.25,30))

# Get AAPL option data
ticker = yf.Ticker("AAPL")
x = ticker.history(period="1d")['Close'].iloc[-1]  # Spot price
expiration = ticker.options[0]
option_chain = ticker.option_chain(expiration)
calls = option_chain.calls

# Time to maturity
T_days = (np.datetime64(expiration) - np.datetime64('today')).astype(int)
tt = T_days / 365

# Strike prices and market prices
K = calls['strike']
market_prices = calls['lastPrice']


# Black-Scholes model prices (with fixed vol)

bs = [black_scholes_model(x, 0, c, 0.05, 0.25, tt) for c in K]

# Plot: Market vs BSM prices
plt.figure(figsize=(10,6))
plt.plot(K, market_prices, label="Market Price", marker='o')
plt.plot(K, bs, label="Black-Scholes Price (σ=25%)", marker='x')
plt.axvline(x, color='gray', linestyle='--', label='Spot Price')
plt.title(f"AAPL Call Option Prices (Exp: {expiration})")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.show()


# Pricing Error Plot
errors = np.array(bs) - np.array(market_prices)
plt.figure(figsize=(10,5))
plt.plot(K, errors, label="Model Error (BS - Market)")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Strike Price")
plt.ylabel("Error ($)")
plt.title("Black-Scholes Pricing Error")
plt.grid(True)
plt.legend()
plt.show()