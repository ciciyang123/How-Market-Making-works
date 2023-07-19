import copy
from matplotlib.ticker import FuncFormatter, MaxNLocator
from numpy import exp, arange, zeros, round
from strategies.baseStrategy import *
from gateway.dataFormat import *
from utils.utility import get_trading_seconds
from utils.pricing.impliedVolatility import ImpliedVolatility, IVSyntheticMethod, IVMethod
from py_vollib_vectorized.api import vectorized_black
from utils.pricing.SVI import StochasticVolatilityInspired
from strategies.utility import *
from matplotlib import pyplot as plt
from scipy.optimize import minimize


class MarketMakerStrategy(BaseStrategy):
    """"""

    """parameters"""
    unit: int = 0
    call_r: float = 0.
    put_r: float = 0.
    year: float = 244.
    max_iv_gap: float = 0.01
    otm_iv_weight: float = 0.7

    """variables"""
    theo_iv: array = array([])
    svi_iv: array = array([])
    weighted_iv: array = array([])
    weights: array = array([])
    spot: float = 0.
    forward: float = 0.
    expire: float = 0.
    strikes: array = array([])
    call_theo: array = array([])
    put_theo: array = array([])
    call_model: array = array([])
    put_model: array = array([])
    bid_vol: array = array([])
    ask_vol: array = array([])
    call_ask: array = array([])
    call_bid: array = array([])
    put_ask: array = array([])
    put_bid: array = array([])
    theo_spreads: dict = dict()
    model_spreads: dict = dict()
    spreads: dict = dict()
    positions: dict = dict()
    open_orders: list = list()
    option_greeks: DataFrame = DataFrame()
    trade_blotters: DataFrame = DataFrame()
    tb_summary: DataFrame = DataFrame()
    option_chain: DataFrame = DataFrame()
    dollar_greeks: DataFrame = DataFrame()
    dollar_greeks_summary: Series = Series()

    parameters = ["unit", "call_r", "put_r", "year", "max_iv_gap", "otm_iv_weight"]
    variables = ["theo_iv", "svi_iv", "weighted_iv", "weights", "spot", "forward", "syn_fwd_price", "strikes",
                 "expire", "spreads", "theo_spreads", "model_spreads", "bid_vol", "ask_vol", "trade_blotters",
                 "option_chain", "tb_summary", "call_theo", "put_theo", "call_model", "put_model", "call_ask",
                 "call_bid", "put_ask", "put_bid",
                 "option_greeks", "positions", "open_orders", "dollar_greeks", "dollar_greeks_summary"]

    def __init__(self, main_engine: MainEngine, strategy_params: StrategyParams):
        #inherit the attributes and methods from baseStrategy and get the keys' values from strategy_params
        super().__init__(main_engine, strategy_params)
        for key in self.strategy_params.__dict__:
            if key in self.parameters:
                self.__setattr__(key, getattr(strategy_params, key))

        self.quotes = self.main_engine.quotes
        self.mkt_data = self.main_engine.mkt_data
        self.iv = ImpliedVolatility(strategy_params)
        self.svi = StochasticVolatilityInspired(strategy_params)

        self.action_space = ActionSpace()
        self.reward_function = RewardFunction()

        self.fig = plt.figure(figsize=(12, 8))
        self.tick_time_list = []
        self.spot_list = []
        self.pnl_list = []
        self.hedger_candidates = DataFrame()

    def __set_variables(self, tick_data: TickData):
        self.expire = tick_data.expire
        self.spot = tick_data.spot
        self.open_orders = self.quotes.get_open_orders()
        self.positions = self.quotes.get_position()

        self.iv(tick_data=tick_data, fwd_method=IVSyntheticMethod.QWinVol, iv_method=IVMethod.QWin)
        self.call_r, self.put_r = self.iv.get_call_put_r()
        self.theo_iv = self.iv.get_iv_theo()
        self.option_greeks = self.iv.get_greeks()
        self.forward = self.iv.forward
        self.strikes = self.iv.strikes
        self.atm_index = self.iv.atm_index

        self.trade_blotters = self.quotes.get_trade_blotters()
        self.option_chain = self.mkt_data.get_option_chain()

        self.action_space.set_variables(n_strikes=len(self.strikes), atm_index=self.atm_index, unit=1e-3)

        self.dollar_greeks = self.option_greeks.copy()

        greeks = ['$delta', '1%$gamma', '$vega', '$theta', '$rho', '$vomma', '$vanna']
        # $Delta 标的资产价格变动的百分比带来的期权价格变动
        self.dollar_greeks['$delta'] = self.dollar_greeks['delta'] * self.forward * self.unit
        # 1%$Gamma 标的资产价格变动1%的平方带来的期权价格变动
        self.dollar_greeks['1%$gamma'] = self.dollar_greeks['gamma'] * self.forward ** 2 * 0.01 * self.unit
        # $Vega 市场隐含波动率变化1%(绝对值而不是相对值)带来的组合净值变化
        self.dollar_greeks['$vega'] = self.dollar_greeks['vega'] * self.unit
        # $Theta 经过1个交易日后组合净值变化
        self.dollar_greeks['$theta'] = self.dollar_greeks['theta'] * 365 / self.year * self.unit
        # $Rho 无风险利率变化1%(绝对值而不是相对值)带来的组合净值变化
        self.dollar_greeks['$rho'] = self.dollar_greeks['rho'] * 0.01 * self.unit
        # $Vomma 市场隐含波动率变化1%(绝对值而不是相对值)的平方带来的组合净值变化
        self.dollar_greeks['$vomma'] = self.dollar_greeks['vomma'] * 0.0001
        # $Vanna 标的资产价格变动的百分比带来的市场隐含波动率变化(绝对值而不是相对值)而引起的组合净值变化
        self.dollar_greeks['$vanna'] = self.dollar_greeks['vanna'] * self.forward * 0.01
        self.option_greeks = self.dollar_greeks.copy()

        self.dollar_greeks['long_pos'] = self.dollar_greeks['option_code'].apply(lambda option_code:
                                                                                 self.positions[option_code].long_pos)
        self.dollar_greeks['short_pos'] = self.dollar_greeks['option_code'].apply(lambda option_code:
                                                                                  self.positions[option_code].short_pos)
        self.dollar_greeks['pos'] = self.dollar_greeks['option_code'].apply(lambda option_code:
                                                                            self.positions[option_code].pos)
        for greek in greeks:
            self.dollar_greeks[greek] *= self.dollar_greeks['pos']
        self.dollar_greeks.drop(['delta', 'gamma', 'vega', 'theta', 'rho', 'vomma', 'vanna'], axis=1, inplace=True)
        self.dollar_greeks_summary = self.dollar_greeks[greeks].sum()

    def __vol_svi(self):
        vol = self.theo_iv
        strikes = self.strikes
        expire = self.expire
        forward = self.forward
        spot = self.spot
        self.svi(vol=vol, strikes=strikes, expire=expire, forward=forward, spot=spot)
        self.svi_iv = self.svi.vol_svi

    def __cal_price(self, iv: array, type_: str):
        self.__setattr__(f"call_{type_}", vectorized_black(
            flag='c', F=self.forward, K=self.strikes, t=self.expire, r=self.call_r, sigma=iv, return_as='numpy')
                         * exp(-self.call_r * self.expire))
        self.__setattr__(f"put_{type_}", vectorized_black(
            flag='p', F=self.forward, K=self.strikes, t=self.expire, r=self.put_r, sigma=iv, return_as='numpy')
                         * exp(-self.put_r * self.expire))
        call_spreads = eval(f"(self.iv.call_ask-self.iv.call_bid)/self.call_{type_}")
        put_spreads = eval(f"(self.iv.put_ask-self.iv.put_bid)/self.put_{type_}")
        self.__setattr__(f"{type_}_spreads", {'call_spreads': call_spreads, 'put_spreads': put_spreads})

    def __trade_blotters_summary(self, tick_data: TickData):
        call = tick_data.call
        put = tick_data.put
        call['theo_price'] = self.call_theo
        put['theo_price'] = self.put_theo
        call.set_index('option_code', inplace=True)
        put.set_index('option_code', inplace=True)

        self.tb_summary = self.trade_blotters.copy()
        self.tb_summary['direction'] = self.tb_summary['side'].map({OrderSide.BUY: 1, OrderSide.SELL: -1})
        self.tb_summary['flagged_pos'] = self.tb_summary['direction'] * self.tb_summary['quantity']
        self.tb_summary['unit'] = self.unit
        self.tb_summary['last_price'] = self.tb_summary['option_code'].apply(
            lambda x: call.at[x, 'last_price'] if x in call.index else put.at[x, 'last_price'])
        self.tb_summary['mkt_pnl'] = (self.tb_summary['last_price'] - self.tb_summary['price']) * \
                                      self.tb_summary['unit'] * self.tb_summary['flagged_pos']

        self.tb_summary['theo_price'] = self.tb_summary['option_code'].apply(
            lambda x: call.at[x, 'theo_price'] if x in call.index else put.at[x, 'theo_price'])
        self.tb_summary['theo_pnl'] = (self.tb_summary['theo_price'] - self.tb_summary['price']) * \
                                       self.tb_summary['unit'] * self.tb_summary['flagged_pos']
        self.tb_summary = self.tb_summary[['commission', 'theo_pnl', 'mkt_pnl']].sum()

    def __hedging_error(self, x):
        delta_vega_converter = 10000.0
        delta_err = (self.dollar_greeks_summary['$delta'] + (x * self.hedger_candidates['$delta']).sum()) ** 2
        vega_err = (self.dollar_greeks_summary['$vega'] + (x * self.hedger_candidates['$vega']).sum()) ** 2
        err = delta_err + vega_err * delta_vega_converter + (x**2).sum()
        self.err = err
        return err

    def hedger(self):
        self.hedger_candidates = self.option_greeks.copy()
        self.hedger_candidates = self.hedger_candidates[
            (self.hedger_candidates['delta'].abs() > 0.2) & (self.hedger_candidates['delta'].abs() < 0.8)]
        self.hedger_candidates.reset_index(drop=True, inplace=True)
        option_num = len(self.hedger_candidates)
        hedger_weights = zeros(option_num)
        res = minimize(self.__hedging_error, method='SLSQP', x0=hedger_weights)
        self.hedger_candidates['weights'] = round(res.x)
        for i in range(option_num):
            option_code = self.hedger_candidates.at[i, 'option_code']
            order_quantity = int(self.hedger_candidates.at[i, 'weights'])
            if order_quantity > 0:
                self.quotes.buy_open(option_code, price=self.mkt_data.get_option_quotes_via_code(option_code, 'ask'),
                                     quantity=order_quantity, order_type=OrderType.LIMIT)
            elif order_quantity < 0:
                self.quotes.sell_open(option_code, price=self.mkt_data.get_option_quotes_via_code(option_code, 'bid'),
                                      quantity=-order_quantity, order_type=OrderType.LIMIT)

    def market_making(self, action: int):
        action = self.action_space.get_action(action)
        self.quotes.cancel_all_orders()
        if abs(self.dollar_greeks_summary['$delta']) > 300000 or abs(self.dollar_greeks_summary['$vega']) > 10000:
            self.hedger()
        else:
            units_bias = action
            n = len(self.strikes)
            self.bid_vol = self.theo_iv + units_bias['bid']
            self.ask_vol = self.theo_iv + units_bias['ask']
            self.call_ask = vectorized_black(flag='c', F=self.forward, K=self.strikes, t=self.expire, r=self.call_r,
                                             sigma=self.ask_vol, return_as='numpy') * exp(-self.call_r * self.expire)
            self.call_bid = vectorized_black(flag='c', F=self.forward, K=self.strikes, t=self.expire, r=self.call_r,
                                             sigma=self.bid_vol, return_as='numpy') * exp(-self.call_r * self.expire)
            self.put_ask = vectorized_black(flag='p', F=self.forward, K=self.strikes, t=self.expire, r=self.put_r,
                                            sigma=self.ask_vol, return_as='numpy') * exp(-self.put_r * self.expire)
            self.put_bid = vectorized_black(flag='p', F=self.forward, K=self.strikes, t=self.expire, r=self.put_r,
                                            sigma=self.bid_vol, return_as='numpy') * exp(-self.put_r * self.expire)
            call_option_codes = self.iv.call_option_code.tolist()
            put_option_codes = self.iv.put_option_code.tolist()
            k = (n-5) // 2
            for i in range(n):
                if i <= k:
                    self.call_bid[i] -= 5
                    self.call_ask[i] += 5
                elif i >= n-k:
                    self.put_bid[i] -= 5
                    self.put_ask[i] += 5
                self.quotes.buy_open(call_option_codes[i], price=round(self.call_bid[i], 4),
                                     quantity=10, order_type=OrderType.LIMIT)
                self.quotes.sell_open(call_option_codes[i], price=round(self.call_ask[i], 4),
                                      quantity=10, order_type=OrderType.LIMIT)
                self.quotes.buy_open(put_option_codes[i], price=round(self.put_bid[i], 4),
                                     quantity=10, order_type=OrderType.LIMIT)
                self.quotes.sell_open(put_option_codes[i], price=round(self.put_ask[i], 4),
                                      quantity=10, order_type=OrderType.LIMIT)

    def on_ticker(self, tick_data):
        tick_data = copy.deepcopy(tick_data)
        tick_data.expire = (tick_data.expire + 1 -
                            get_trading_seconds(tick_time=tick_data.tick_time) / (4 * 60 * 60)) / self.year
        self.__set_variables(tick_data)
        self.__vol_svi()
        self.__cal_price(iv=self.theo_iv, type_="theo")
        self.__cal_price(iv=self.svi_iv, type_="model")
        self.__trade_blotters_summary(tick_data)
        self.spreads = self.theo_spreads

        action = 0
        self.market_making(action)
        self.reward_function.update_reward(self.tb_summary['theo_pnl'])
        self.action_space.update_action(action)
        self.plot_positions(tick_time=tick_data.tick_time, action=action)

    def plot_positions(self, tick_time, action):
        def format_fn(tick_val, tick_pos):
            if int(tick_val) in xs:
                return self.tick_time_list[int(tick_val)]
            else:
                return ''

        t = tick_time.strftime("%Y-%m-%d %H:%M:%S")
        self.fig.suptitle(f'{t}  Cumulative PnL: {int(self.reward_function.cum_reward)}')
        layout = (3, 2)

        self.spot_list.append(self.spot)
        self.tick_time_list.append(t.split(" ")[1])
        self.pnl_list.append(int(self.reward_function.cum_reward))
        xs = range(len(self.tick_time_list))

        spot_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        spot_ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
        spot_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pnl_ax = spot_ax.twinx()
        spot_ax.plot(xs, self.spot_list, color='red', label='spot')
        pnl_ax.plot(xs, self.pnl_list, color='green', label='PnL')
        spot_ax.grid()
        spot_ax.legend(loc='upper left')
        pnl_ax.legend(loc='upper right')

        raw_iv_ax = plt.subplot2grid(layout, (1, 0))
        raw_iv_ax.plot(self.strikes, self.iv.iv_call_ask, 'm.-.', color='red', label='call_ask')
        raw_iv_ax.plot(self.strikes, self.iv.iv_call_bid, 'm.-.', color='green', label='call_bid')
        raw_iv_ax.plot(self.strikes, self.iv.iv_put_ask, 'm.-.', color='yellow', label='put_ask')
        raw_iv_ax.plot(self.strikes, self.iv.iv_put_bid, 'm.-.', color='blue', label='put_bid')
        raw_iv_ax.plot(self.strikes, self.theo_iv, label='theo_iv', linewidth=2)
        raw_iv_ax.grid()
        raw_iv_ax.set_title(f"Call r: {round(self.call_r*100, 2)}%  Put r: {round(self.put_r*100, 2)}%")
        raw_iv_ax.legend(loc='upper center')

        order_iv_ax = plt.subplot2grid(layout, (1, 1), sharey=raw_iv_ax)
        order_iv_ax.plot(self.strikes, self.bid_vol, 'm.-.', color='red', label='bid_vol')
        order_iv_ax.plot(self.strikes, self.ask_vol, 'm.-.', color='green', label='ask_vol')
        order_iv_ax.plot(self.strikes, self.theo_iv, label='theo_iv', linewidth=2)
        order_iv_ax.plot(self.strikes, self.svi_iv, label=r'SVI $a$: ' + f'{round(self.svi.a, 4)}, ' +
                                                          r'$b$: ' + f'{round(self.svi.b, 4)}, ' +
                                                          r'$m$: ' + f'{round(self.svi.m, 4)}, ' +
                                                          r'$\rho$: ' + f'{round(self.svi.rho, 4)}, ' +
                                                          r'$\sigma$: ' + f'{round(self.svi.sigma, 4)} ', linewidth=2)
        order_iv_ax.grid()
        order_iv_ax.set_title(f"order iv: action={action}")
        order_iv_ax.legend(loc='upper center')

        position_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
        bar_width = 0.3
        n = len(self.strikes)
        index_call = arange(n)
        index_put = index_call + bar_width
        long_pos = self.dollar_greeks['long_pos']
        short_pos = self.dollar_greeks['short_pos']
        deltas = self.dollar_greeks['$delta']
        vegas = self.dollar_greeks['$vega']
        position_ax.bar(index_call, height=long_pos[:n], width=bar_width, label='long call')
        position_ax.bar(index_call, height=-short_pos[:n], width=bar_width, label='short call')
        position_ax.bar(index_put, height=long_pos[n:], width=bar_width, label='long put')
        position_ax.bar(index_put, height=-short_pos[n:], width=bar_width, label='short put')
        position_ax.legend()
        xtick_labels = [f'{self.strikes[i]}' \
                        f'\n{"%.2g" % (deltas[i] + deltas[i + n])}' \
                        f'\n{"%.2g" % (vegas[i] + vegas[i + n])}' for i in range(n)]
        position_ax.set_xticks(index_call + bar_width / 2)
        position_ax.set_xticklabels(xtick_labels)
        position_ax.set_title(f"Delta: {round(self.dollar_greeks_summary['$delta'], 2)}  "
                              f"Gamma: {round(self.dollar_greeks_summary['1%$gamma'], 2)}  "
                              f"Vega: {round(self.dollar_greeks_summary['$vega'], 2)}  "
                              f"Theta: {round(self.dollar_greeks_summary['$theta'], 2)}  "
                              f"Rho: {round(self.dollar_greeks_summary['$rho'], 2)}  "
                              f"Vomma: {round(self.dollar_greeks_summary['$vomma'], 2)}  "
                              f"Vanna: {round(self.dollar_greeks_summary['$vanna'], 2)}  ")
        position_ax.grid()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

        raw_iv_ax.cla()
        order_iv_ax.cla()
        position_ax.cla()
