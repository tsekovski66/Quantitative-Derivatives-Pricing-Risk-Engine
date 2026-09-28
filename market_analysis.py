import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from options_pricing import implied_volatility

def get_live_risk_free_rate():
    """Pulls the live 13-week US Treasury Bill yield (^IRX) to use as risk-free rate"""
    try:
        # ^IRX is quoted as a whole percentage, divide by 100
        live_rate = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        return live_rate
    except Exception:
        # Safeguard if the data pull fails
        return 0.04

def fetch_and_clean_data(ticker_symbol, risk_free_rate):
    """Pulls live market data"""
    ticker = yf.Ticker(ticker_symbol)
    spot_price = ticker.history(period="1d")['Close'].iloc[-1]
    
    # Target the 4th expiration date, avoids 0DTE (Zero Days to Expiration) contracts 
    # Carry gamma noise and distort the baseline volatility skew
    target_expiry = ticker.options[3]
    
    # Annualize the time to maturity (T)
    T = max((datetime.strptime(target_expiry, '%Y-%m-%d') - datetime.today()).days / 365.0, 0.001)
    
    # Pull the live Put options chain
    puts = ticker.option_chain(target_expiry).puts
    
    # Remove anomalies that have zero active bids or asks
    puts = puts[(puts['bid'] > 0) & (puts['ask'] > 0)]
    
    # Isolate the strikes strictly within +/- 20% of the current spot price
    filtered_puts = puts[(puts['strike'] >= spot_price * 0.80) & (puts['strike'] <= spot_price * 1.20)].copy()
    
    # Use the bid-ask midpoint, avoiding stale 'last price' prints
    filtered_puts['market_price'] = (filtered_puts['bid'] + filtered_puts['ask']) / 2
    
    return filtered_puts, spot_price, T, target_expiry

def generate_volatility_smile(ticker_symbol="NVDA"):
    """Extracts IV and plots the market skew U-curve to expose institutional tail-risk hedging"""
    dynamic_rfr = get_live_risk_free_rate()
    data, spot, T, expiry = fetch_and_clean_data(ticker_symbol, dynamic_rfr)
    
    iv_results, valid_strikes = [], []
    
    # Run the SciPy solver
    for index, row in data.iterrows():
        iv = implied_volatility(row['market_price'], spot, row['strike'], T, dynamic_rfr, option_type="put")
        
        # Append only successful convergences
        if iv is not None:
            iv_results.append(iv * 100) # Convert to percentage
            valid_strikes.append(row['strike'])
            
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    # Plot the smile
    plt.plot(valid_strikes, iv_results, marker='o', linestyle='-', color='cyan', linewidth=2, markersize=6)
    
    # Mark the underlying asset's current price
    plt.axvline(x=spot, color='red', linestyle='--', label=f'Spot Price (${spot:.2f})')
    
    plt.title(f'{ticker_symbol} Volatility Skew (Expiry: {expiry}) | RFR: {dynamic_rfr*100:.2f}%', fontsize=14, fontweight='bold')
    plt.xlabel('Strike Price ($)', fontsize=12)
    plt.ylabel('Implied Volatility (%)', fontsize=12)
    plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_volatility_smile("NVDA")
    generate_volatility_smile("AAPL")
    generate_volatility_smile("TSLA")
    generate_volatility_smile("MSFT")
    generate_volatility_smile("AMZN")
    generate_volatility_smile("GOOGL")
    generate_volatility_smile("META")
    generate_volatility_smile("AMD")
    generate_volatility_smile("INTC")
    generate_volatility_smile("NFLX")