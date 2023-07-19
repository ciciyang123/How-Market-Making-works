from scipy.optimize import minimize
from strategies.strategyParams import StrategyParams
from numpy import array, exp, min, max, where, isnan, nan, nanmedian, sqrt, pi, log
from pandas import concat, DataFrame
from enum import Enum
from py_vollib.black import implied_volatility
from gateway.dataFormat import TickData
from utils.pricing.pricingEngine import PricingEngineError
from py_vollib.black.greeks import analytical
from py_vollib.ref_python.black import d1


def implied_vol(price: float, f: float, k: float, r: float, t: float, call_put: str) -> float:
    try:
        vol = implied_volatility.implied_volatility_of_discounted_option_price(
            discounted_option_price=price, F=f, K=k, r=r, t=t, flag=call_put)
    except:
        return nan
    if vol < 0:
        raise PricingEngineError(
            state=PricingEngineError.State.NegativeVolatility
        )
    return vol


def implied_vol_decorator(iv_func):
    def wrapper(*args, **kwargs):
        ia = nan
        ib = nan
        if isnan(kwargs['pa']) or isnan(kwargs['pb']):
            pass
        else:
            try:
                ia, ib = iv_func(*args, **kwargs)
            except PricingEngineError as e:
                if e.state_ == PricingEngineError.State.NegativeVolatility:
                    pass
                else:
                    raise e
        return ia, ib
    return wrapper


@implied_vol_decorator
def black_implied_vol(pa, pb, forward, k, t, r, call_put):
    ia = implied_vol(price=pa, f=forward, k=k, r=r, t=t, call_put=call_put)
    ib = implied_vol(price=pb, f=forward, k=k, r=r, t=t, call_put=call_put)
    return ia, ib


@implied_vol_decorator
def qwin_implied_vol(pa, pb, fwd_bid, fwd_ask, k, r, t, call_put):
    if call_put == 'c':
        ia = implied_vol(price=pa, f=fwd_bid, k=k, r=r, t=t, call_put='c')
        ib = implied_vol(price=pb, f=fwd_ask, k=k, r=r, t=t, call_put='c')
    else:
        ia = implied_vol(price=pa, f=fwd_ask, k=k, r=r, t=t, call_put='p')
        ib = implied_vol(price=pb, f=fwd_bid, k=k, r=r, t=t, call_put='p')
    return ia, ib


class IVSyntheticMethod(Enum):
    MedianFwdVol = "MedianFwdVol"
    AtmFwdVol = "AtmFwdVol"
    QWinVol = "QWinVol"
    PutCallMove = "PutCallMove"
    PutCallMeanMove = "PutCallMeanMove"


class IVMethod(Enum):
    QWin = "QWin"
    Black = "Black"


class InterestRateCalibrator:

    def __init__(self):
        self.strikes = array([])
        self.forward = 0.
        self.fwd_ask = 0.
        self.fwd_bid = 0.
        self.expire = 0.
        self.call_ask = array([])
        self.call_bid = array([])
        self.put_ask = array([])
        self.put_bid = array([])
        self.k = array([])
        self.weights = array([])
        self.call_r = 0.
        self.put_r = 0.
        self.max_r = 0.
        self.err = 1e3
        self.iv_method = None

    def set_variables(self,
                      strikes: array,
                      forward: float,
                      fwd_ask: float,
                      fwd_bid: float,
                      expire: float,
                      call_ask: array,
                      call_bid: array,
                      put_ask: array,
                      put_bid: array,
                      iv_method: IVMethod,
                      call_r: float = 0.,
                      put_r: float = 0.,
                      max_r: float = 0.05,
                      ):
        self.strikes = strikes
        self.forward = forward
        self.fwd_ask = fwd_ask
        self.fwd_bid = fwd_bid
        self.expire = expire
        self.call_ask = call_ask
        self.call_bid = call_bid
        self.put_ask = put_ask
        self.put_bid = put_bid
        self.k = log(strikes / forward)
        self.iv_method = iv_method
        self.call_r = call_r
        self.put_r = put_r
        self.max_r = max_r
        self.weights = 1/(abs(self.k)+0.05)

    def __con(self):
        # x0: call_r
        # x1: put_r
        cons = (
            # 0 < call_r < max_r
            {'type': 'ineq', 'fun': lambda x: x[0] - 0},
            {'type': 'ineq', 'fun': lambda x: self.max_r - x[0]},
            # 0 < put_r < max_r
            {'type': 'ineq', 'fun': lambda x: x[1] - 0},
            {'type': 'ineq', 'fun': lambda x: self.max_r - x[1]},
        )
        return cons

    def __error(self, x):
        self.call_r, self.put_r = x
        err = 0.
        n = len(self.strikes)
        for i in range(0, n):
            k = self.strikes[i]
            # calculate the call implied volatility
            cpa = self.call_ask[i]
            cpb = self.call_bid[i]
            cia = cib = nan
            if self.iv_method == IVMethod.QWin:
                cia, cib = qwin_implied_vol(pa=cpa, pb=cpb, fwd_ask=self.fwd_ask, fwd_bid=self.fwd_bid,
                                            k=k, r=self.call_r, t=self.expire, call_put='c')
            elif self.iv_method == IVMethod.Black:
                cia, cib = black_implied_vol(pa=cpa, pb=cpb, forward=self.forward,
                                             k=k, r=self.call_r, t=self.expire, call_put='c')
            # calculate the put implied volatility
            ppa = self.put_ask[i]
            ppb = self.put_bid[i]
            pia = pib = nan
            if self.iv_method == IVMethod.QWin:
                pia, pib = qwin_implied_vol(pa=ppa, pb=ppb, fwd_ask=self.fwd_ask, fwd_bid=self.fwd_bid,
                                            k=k, r=self.put_r, t=self.expire, call_put='p')
            elif self.iv_method == IVMethod.Black:
                pia, pib = black_implied_vol(pa=ppa, pb=ppb, forward=self.forward,
                                             k=k, r=self.put_r, t=self.expire, call_put='p')

            if (not isnan(pia)) and (not isnan(pib)) and (not isnan(cia)) and (not isnan(cib)):
                cim = (cia + cib) / 2
                pim = (pia + pib) / 2
                # MAE
                err += abs(cim-pim)*self.weights[i]
        self.err = err
        return err

    def calibrate(self):
        x0 = array([self.call_r, self.put_r])  # a, b, m, rho, sigma
        self.err = self.__error(x0)
        if self.err >= 0.15:
            cons = self.__con()
            res = minimize(self.__error, method='SLSQP', x0=x0, constraints=cons)
            self.call_r, self.put_r = res.x
        return self.call_r, self.put_r


class ImpliedVolatility:
    r_ = 0.
    call_r = 0.
    put_r = 0.
    max_iv_gap = 0.01
    otm_iv_weight = 0.7

    parameters = ["call_r", "put_r", "max_iv_gap", "otm_iv_weight"]

    # constructor
    def __init__(self, strategy_params: StrategyParams):

        self.strategy_params = strategy_params
        for key in self.strategy_params.__dict__:
            if key in self.parameters:
                self.__setattr__(key, getattr(strategy_params, key))

        self.expire = 0.0
        self.spot = 0.0
        self.spot_ask = 0.0
        self.spot_bid = 0.0
        self.fwd_asks = 0.0
        self.fwd_bids = 0.0
        self.forward = 0.0
        self.atm_index_strikes = 0.0
        self.strikes = array([])

        self.call_ask = array([])
        self.call_bid = array([])
        self.put_ask = array([])
        self.put_bid = array([])

        self.iv_call_ask = array([])
        self.iv_call_bid = array([])
        self.iv_put_ask = array([])
        self.iv_put_bid = array([])
        self.iv_call_mid = array([])
        self.iv_put_mid = array([])

        self.calibrator = InterestRateCalibrator()

        self.last_valid_iv_theo = array([])
        self.iv_theo = array([])
        self.call_theo = array([])
        self.put_theo = array([])
        self.greeks = DataFrame()

    def get_iv_call_ask(self):
        return self.iv_call_ask

    def get_iv_call_bid(self):
        return self.iv_call_bid

    def get_iv_put_ask(self):
        return self.iv_put_ask

    def get_iv_put_bid(self):
        return self.iv_put_bid

    def get_call_put_r(self):
        return self.call_r, self.put_r

    def get_last_valid_iv_theo(self):
        if not len(self.last_valid_iv_theo):
            self.last_valid_iv_theo = [nan] * len(self.strikes)
        return self.last_valid_iv_theo

    def get_iv_theo(self):
        return self.iv_theo

    def get_greeks(self):
        return self.greeks

    def __iv(self, iv_method: IVMethod):

        valid = True
        n = len(self.strikes)
        iv_call_ask = list()
        iv_call_bid = list()
        iv_put_ask = list()
        iv_put_bid = list()
        iv_call_mid = list()
        iv_put_mid = list()
        iv_theo = list()

        for i in range(0, n):
            k = self.strikes[i]
            # calculate the call implied volatility
            cpa = self.call_ask[i]
            cpb = self.call_bid[i]
            cia = cib = nan
            if iv_method == IVMethod.QWin:
                cia, cib = qwin_implied_vol(pa=cpa, pb=cpb, fwd_ask=self.fwd_ask, fwd_bid=self.fwd_bid,
                                            k=k, r=self.call_r, t=self.expire, call_put='c')
            elif iv_method == IVMethod.Black:
                cia, cib = black_implied_vol(pa=cpa, pb=cpb, forward=self.forward,
                                             k=k, r=self.call_r, t=self.expire, call_put='c')
            iv_call_ask.append(cia)
            iv_call_bid.append(cib)
            cim = nan
            if (not isnan(cia)) and (not isnan(cib)) and (cia - cib <= self.max_iv_gap):
                cim = (cia + cib) / 2
            else:
                valid = False if k == self.atm_index_strikes else True
            iv_call_mid.append(cim)
            # calculate the put implied volatility
            ppa = self.put_ask[i]
            ppb = self.put_bid[i]
            pia = pib = nan
            if iv_method == IVMethod.QWin:
                pia, pib = qwin_implied_vol(pa=ppa, pb=ppb, fwd_ask=self.fwd_ask, fwd_bid=self.fwd_bid,
                                            k=k, r=self.put_r, t=self.expire, call_put='p')
            elif iv_method == IVMethod.Black:
                pia, pib = black_implied_vol(pa=ppa, pb=ppb, forward=self.forward,
                                             k=k, r=self.put_r, t=self.expire, call_put='p')
            iv_put_ask.append(pia)
            iv_put_bid.append(pib)
            pim = nan
            if (not isnan(pia)) and (not isnan(pib)) and (pia - pib <= self.max_iv_gap):
                pim = (pia + pib) / 2
            elif k == self.atm_index_strikes:
                valid = False if k == self.atm_index_strikes else True
            iv_put_mid.append(pim)
            # calculate the theoretical implied volatility
            if k <= self.atm_index_strikes:
                theo = pim
                if not isnan(cim):
                    theo = theo * self.otm_iv_weight + (1 - self.otm_iv_weight) * cim
            else:
                theo = cim
                if not isnan(pim):
                    theo = theo * self.otm_iv_weight + (1 - self.otm_iv_weight) * pim
            iv_theo.append(theo)

        self.iv_call_ask = array(iv_call_ask)
        self.iv_call_mid = array(iv_call_mid)
        self.iv_call_bid = array(iv_call_bid)
        self.iv_put_ask = array(iv_put_ask)
        self.iv_put_mid = array(iv_put_mid)
        self.iv_put_bid = array(iv_put_bid)
        self.iv_theo = array(iv_theo)
        return valid

    # solve the implied volatility
    def __call__(
            self, tick_data: TickData, fwd_method: IVSyntheticMethod, iv_method: IVMethod
    ):
        self.__read_tick_data(tick_data=tick_data)
        self.__forward_price(method=fwd_method)
        self.__theo_iv(method=iv_method)
        self.__greeks()

    # read the ETF and option quotation
    def __read_tick_data(self, tick_data: TickData):
        # read the future quotation
        self.tick_time = tick_data.tick_time
        self.spot = tick_data.spot
        self.spot_ask = tick_data.spot_ask
        self.spot_bid = tick_data.spot_bid

        # read the option quotation
        self.expire = tick_data.expire
        call = tick_data.call
        put = tick_data.put
        call_strikes = call.index.values
        put_strikes = put.index.values
        assert ((call_strikes == put_strikes).all())
        self.strikes = call_strikes
        self.call_option_code = call['option_code'].values
        self.put_option_code = put['option_code'].values
        self.call_ask = call['ask'].values
        self.call_bid = call['bid'].values
        self.put_ask = put['ask'].values
        self.put_bid = put['bid'].values

    # solve the synthetic forward price
    def __forward_price(self, method: IVSyntheticMethod):

        if method == IVSyntheticMethod.MedianFwdVol:
            self.fwd_asks = exp(self.r_ * self.expire) * (self.call_ask - self.put_bid) + self.strikes
            self.fwd_bids = exp(self.r_ * self.expire) * (self.call_bid - self.put_ask) + self.strikes
            self.forward = nanmedian((self.fwd_asks + self.fwd_bids) / 2)

        elif method == IVSyntheticMethod.QWinVol:
            self.fwd_asks = (self.call_ask - self.put_bid) + self.strikes
            self.fwd_bids = (self.call_bid - self.put_ask) + self.strikes
            num_syn_fwds = 1
            dist = abs(self.strikes - self.spot)
            indices = dist.argsort()[:num_syn_fwds]
            fwd_asks = self.fwd_asks[indices]
            fwd_bids = self.fwd_bids[indices]
            self.fwd_ask = self.fwd_bid = nan
            if (isnan(fwd_asks).sum() < num_syn_fwds) and (isnan(fwd_bids).sum() < num_syn_fwds):
                self.fwd_ask = nanmedian(fwd_asks)
                self.fwd_bid = nanmedian(fwd_bids)
            self.forward = nanmedian((self.fwd_ask + self.fwd_bid) / 2)
            self.atm_index = indices[0]
            self.atm_index_strikes = self.strikes[indices][0]

        else:
            difference = (self.call_ask + self.call_bid) / 2 - (self.put_ask + self.put_bid) / 2
            # replace nan value with zero
            difference = where(isnan(difference), 0, difference)
            spread = min(difference)
            index = where(difference == spread)
            self.forward = exp(self.r_ * self.expire) * spread + self.strikes[index][0]
            index = where(self.strikes <= self.forward)
            index = where(self.strikes == max(self.strikes[index]))
            self.atm_index_strikes = self.strikes[index][0]

    def __theo_iv(self, method: IVMethod):

        self.calibrator.set_variables(strikes=self.strikes, forward=self.forward, fwd_ask=self.fwd_ask,
                                      fwd_bid=self.fwd_bid, expire=self.expire, call_ask=self.call_ask,
                                      call_bid=self.call_bid, put_ask=self.put_ask, put_bid=self.put_bid,
                                      iv_method=IVMethod.QWin, call_r=self.call_r, put_r=self.put_r)
        self.call_r, self.put_r = self.calibrator.calibrate()
        valid = self.__iv(method)
        if not valid:
            self.iv_theo = self.get_last_valid_iv_theo()
        else:
            self.last_valid_iv_theo = self.th()

    def __greeks(self):
        def cal_greeks(flag: str, F: float, K: float, t: float, r: float, sigma: float):
            D1 = d1(F=F, K=K, t=t, r=r, sigma=sigma)
            greeks = {
                'delta': analytical.delta(flag=flag, F=F, K=K, t=t, r=r, sigma=sigma),
                'gamma': analytical.gamma(flag=flag, F=F, K=K, t=t, r=r, sigma=sigma),
                'vega': analytical.vega(flag=flag, F=F, K=K, t=t, r=r, sigma=sigma),
                'theta': analytical.theta(flag=flag, F=F, K=K, t=t, r=r, sigma=sigma),
                'rho': analytical.rho(flag=flag, F=F, K=K, t=t, r=r, sigma=sigma),
                'vanna': exp(-r * t - D1 ** 2 / 2) * sqrt(t / (2 * pi)) * (1 - D1 / (sigma * sqrt(t))),
                'vomma': -exp(-r * t - D1 ** 2 / 2) * t / 2 * sqrt(1 / (2 * pi)) * F * D1
            }
            return greeks

        n = len(self.strikes)
        call_greeks = dict()
        put_greeks = dict()
        for i in range(n):
            call_greeks[self.call_option_code[i]] = cal_greeks(flag='c', F=self.forward, K=self.strikes[i],
                                                               t=self.expire, r=self.call_r, sigma=self.iv_theo[i])
            put_greeks[self.put_option_code[i]] = cal_greeks(flag='p', F=self.forward, K=self.strikes[i],
                                                             t=self.expire, r=self.put_r, sigma=self.iv_theo[i])
        self.greeks = concat([DataFrame(call_greeks).T, DataFrame(put_greeks).T])
        self.greeks.reset_index(inplace=True)
        self.greeks.rename(columns={'index': 'option_code'}, inplace=True)
