from sympy import *
from numpy import array, sqrt, power, log, abs
import numpy as np
from scipy.optimize import minimize
import py_vollib_vectorized
from strategies.strategyParams import StrategyParams

'''
    Stochastic Volatility Inspired (SVI) Model:
    1) Raw SVI
        w(k, t) = a + b*( rho*(k-m) + sqrt((k-m)**2 + sigma**2) )
            - w(k, t) = sigma(k, t)**2
            - sigma(k, t): implied volatility
            - k: log strike
            - t: expiration time
            - parameter: a, b, rho, m, sigma

    2) Jump Wing SVI
'''


class StochasticVolatilityInspired:

    r = 0.
    q = 0.
    parameters = ["r", "q"]

    def __init__(
            self,
            strategy_params: StrategyParams
    ):

        self.strategy_params = strategy_params
        for key in self.strategy_params.__dict__:
            if key in self.parameters:
                self.__setattr__(key, getattr(strategy_params, key))

        self.expire = 0.
        self.spot = 0.
        self.forward = 0.
        self.vol = array([])
        self.implied_variance = array([])
        self.total_variance = array([])
        self.vol_svi = array([])
        self.strikes = array([])
        self.k = array([])

        self.vega = list()
        self.a = 0.0  # a vertical translation of the smile
        self.b = 0.0  # the slopes of both the put and call wings, tightening the smile
        self.rho = 0.0  # decreases the slope of the left wing, a counter-clockwise rotation of the smile
        self.m = 0.0  # translates the smile to the right
        self.sigma = 0.0  # reduce the at-the-money(ATM) curvature of the smile
        self.err = 0.0

    def omega(self, k) -> float:  # cpu matrix operation
        variance = self.a + self.b * (self.rho * (k - self.m) + sqrt((k - self.m) ** 2 + self.sigma ** 2))
        return variance

    def __price_error(self, x):  # cpu matrix operation
        self.a, self.b, self.m, self.rho, self.sigma = x
        var_svi = self.omega(k=self.k)
        self.vol_svi = np.sqrt(var_svi)
        err = self.implied_variance - var_svi
        # MAE
        err = np.abs(err)
        err = np.multiply(err, self.vega)
        err = np.sum(err)
        self.err = err
        return err

    def __vega(self):  # cpu matrix operation
        n = len(self.strikes)
        flag = ['c'] * n
        s = self.spot
        k = self.strikes
        t = self.expire
        r = self.r
        q = self.q
        sigma = self.vol
        model = 'black_scholes_merton'
        self.vega = py_vollib_vectorized.vectorized_vega(flag=flag, S=s, K=k, t=t, r=r, sigma=sigma, q=q, model=model,
                                                         return_as='numpy')

    def __con(self):
        w = np.max(self.total_variance)
        t = self.expire
        min_k = np.min(self.k)
        max_k = np.max(self.k)
        # x0: a
        # x1: b
        # x2: m
        # x3: rho
        # x4: sigma
        cons = (
            # 0 < a < max(w)
            {'type': 'ineq', 'fun': lambda x: x[0] - 0},
            {'type': 'ineq', 'fun': lambda x: w - x[0]},
            # 0 < b < 4/(t*(1+abs(rho)))
            {'type': 'ineq', 'fun': lambda x: x[1] - 0},
            {'type': 'ineq', 'fun': lambda x: 4 / (1 + abs(x[3])) / t - x[1]},
            # min_k < m < max_k
            {'type': 'ineq', 'fun': lambda x: x[2] - min_k},
            {'type': 'ineq', 'fun': lambda x: max_k - x[2]},
            # abs(rho) < 1
            {'type': 'ineq', 'fun': lambda x: 1 - abs(x[3])},
            # sigma > sigma_min
            {'type': 'ineq', 'fun': lambda x: x[4] - 0.0005},
        )
        return cons

    def __calibration(self):
        self.__vega()
        min_vol = np.min(self.vol)
        min_idx = np.argmin(self.vol)
        a = power(min_vol, 2) / 1.2
        m = self.k[min_idx]
        n = len(self.strikes)
        k0 = m
        k1 = self.k[1]
        k2 = self.k[n - 2]
        sigma1 = self.total_variance[1]
        sigma2 = self.total_variance[n - 2]
        sigma0 = self.total_variance[min_idx]
        b1 = np.abs((sigma0 - sigma1) / (k0 - k1))
        b2 = np.abs((sigma2 - sigma0) / (k2 - k0))
        b = (b1 + b2) / 2
        x0 = array([a, b, m, 0.0, 0.05])  # a, b, m, rho, sigma
        cons = self.__con()
        res = minimize(self.__price_error, method='SLSQP', x0=x0, constraints=cons)
        self.a, self.b, self.rho, self.m, self.sigma = res.x

    def __call__(self,
                 vol: array,
                 strikes: array,
                 spot: float,
                 forward: float,
                 expire: float,
                 ):
        self.expire = expire
        self.spot = spot
        self.forward = forward
        self.vol = vol
        self.implied_variance = np.power(self.vol, 2)
        self.total_variance = np.power(self.vol, 2) * self.expire
        self.vol_svi = vol
        self.strikes = strikes
        self.k = log(strikes / forward)

        self.__calibration()

    @property
    def params(self):
        return {'a': self.a, 'b': self.b, 'rho': self.rho, 'm': self.m, 'sigma': self.sigma}
