import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesEngine:
    """Analytical pricing engine for European options."""
    def __init__(self, S, K, T, r, sigma):
        self.S, self.K, self.T, self.r, self.sigma = float(S), float(K), float(T), float(r), float(sigma)

    def _d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self._d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self._d2()) - self.S * norm.cdf(-self._d1())

class BinomialTreeEngine:
    """Discrete-time binomial matrix for early-exercise American options."""
    def __init__(self, S, K, T, r, sigma, steps=100):
        self.S, self.K, self.T, self.r, self.sigma, self.N = float(S), float(K), float(T), float(r), float(sigma), int(steps)
        
    def price(self, option_type="call", is_american=True):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        discount = np.exp(-self.r * dt)
        
        S_T = self.S * (u ** np.arange(self.N, -1, -1)) * (d ** np.arange(0, self.N + 1, 1))
        V = np.maximum(0, S_T - self.K) if option_type == "call" else np.maximum(0, self.K - S_T)
            
        for i in range(self.N - 1, -1, -1):
            V = discount * (p * V[:-1] + (1 - p) * V[1:])
            if is_american:
                S_t = self.S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1, 1))
                V = np.maximum(V, S_t - self.K) if option_type == "call" else np.maximum(V, self.K - S_t)
        return V[0]

def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """Uses Brent's Method to safely extract Implied Volatility."""
    if option_type == "put" and market_price < (K * np.exp(-r * T) - S):
        return None 
    elif option_type == "call" and market_price < (S - K * np.exp(-r * T)):
        return None

    def objective_function(sigma):
        engine = BlackScholesEngine(S, K, T, r, sigma)
        return engine.call_price() - market_price if option_type == "call" else engine.put_price() - market_price

    try:
        return brentq(objective_function, 1e-4, 5.0)
    except (ValueError, RuntimeError):
        return None