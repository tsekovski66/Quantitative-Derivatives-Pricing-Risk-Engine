# Quantitative Derivatives Pricing & Risk Engine

An institutional-grade, object-oriented Python derivatives library bridging continuous-time academic calculus with live market microstructure. This project calculates theoretical option fair values, models dynamic risk sensitivities (The Greeks), and utilizes bounded root-finding algorithms to extract live market implied volatility.

By integrating the `yfinance` API, the engine bypasses theoretical assumptions to expose structural market phenomena, specifically targeting the post-1987 institutional skew for out-of-the-money crash protection.

## Core Architecture

The repository is modularized into two primary components to separate core mathematical operations from live data execution pipelines:

### 1. The Math Library (`options_engine.py`)
* **Analytical Pricer (Black-Scholes):** Calculates the continuous-time theoretical fair value of European Call and Put options
* **Discrete-Time Numerical Pricer (Binomial Tree):** Utilizes a vectorized Cox-Ross-Rubinstein (CRR) NumPy matrix to price American options, capturing the early-exercise premium via backward induction
* **Risk Sensitivities (The Greeks):** Calculates first and second-order partial derivatives ($\Delta, \Gamma, \Theta, \nu$) to measure dynamic portfolio exposure to directional price, time decay, and volatility expansion
* **Root-Finding Solver (Brent's Method):** Replaces standard Newton-Raphson approximation with `SciPy brentq` to mathematically bound volatility extraction, preventing algorithmic divergence when encountering arbitrage violations or near-zero Vega

### 2. The Execution Pipeline (`market_analysis.py`)
* **Live Market Data Integration:** Fetches real-time equity spot prices and options chains via the `yfinance` API
* **Dynamic Risk-Free Rate:** Automatically scrapes the live 13-week US Treasury Bill yield (`^IRX`) to establish an accurate baseline discount rate
* **Microstructure Sanitization:** Filters out 0DTE noise, isolates strikes within +/- 20% of the spot price, and executes pricing using bid-ask midpoints rather than stale "last traded" prints
* **Visualization:** Renders the Implied Volatility U-Curve (The Volatility Smile) using `matplotlib`
