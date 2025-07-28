import matplotlib.pyplot as plt 
import numpy as np 

STRIKE_PRICE = 100
STOCK_PRICE = np.arange(0, 1.1 * 400) 
PREMIUM = 20 

call_buyer = np.maximum(0, STOCK_PRICE - STRIKE_PRICE)
profit_call_buyer = call_buyer - PREMIUM
profit_call_seller = -profit_call_buyer


put_buyer = np.maximum(0, STRIKE_PRICE - STOCK_PRICE)
profit_put_buyer = put_buyer - PREMIUM
profit_put_seller = -profit_put_buyer


plt.plot(STOCK_PRICE, profit_call_buyer, label="Call Option Buyer")
plt.plot(STOCK_PRICE, profit_call_seller, label="Call Option Seller", linestyle='--')

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(STRIKE_PRICE, color='red', linestyle='--', label='Strike Price')

plt.title("Profit or Loss for Call Option")
plt.xlabel("Stock Price at Expiration")
plt.ylabel("Profit or Loss")
plt.grid(True)
plt.legend()
plt.show()


plt.plot(STOCK_PRICE, profit_put_buyer, label="Put Option Buyer")
plt.plot(STOCK_PRICE, profit_put_seller, label="Put Option Seller", linestyle='--')

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(STRIKE_PRICE, color='red', linestyle='--', label='Strike Price')

plt.title("Profit or Loss for Put Option")
plt.xlabel("Stock Price at Expiration")
plt.ylabel("Profit or Loss")
plt.grid(True)
plt.legend()
plt.show()