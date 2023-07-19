import os
from exchange.TickQuotation.TickQuotation import TickQuotation
from exchange.TickQuotation.ETFOptionQuotation import ETFOptionTick
from exchange.utility import *
from strategies.strategyParams import StrategyParams
from utils.event.event import *
from utils.event.eventEngine import EventEngine
from gateway.dataFormat import *
from pandas import to_datetime
from numpy import random, nan, unique, array, linspace, zeros


class SingleOptionPosition:

    def __init__(self, strike: float, option_type: str):

        self.strike = strike
        self.type = "call" if option_type == "认购" else "put"
        self.long_pos = self.short_pos = self.pos = 0
        self.pre_settlement_price = 0
        self.margin = 0

    def __repr__(self):
        return str(self.pos)

    def update(self, quantity: int, side: OrderSide):
        if quantity > 0:
            if side == OrderSide.BUY:
                self.long_pos += quantity
            else:
                self.short_pos += quantity
        elif quantity < 0:
            quantity = -quantity
            if side == OrderSide.BUY:
                if quantity >= self.long_pos:
                    self.long_pos = 0
                    self.update(quantity-self.long_pos, OrderSide.SELL)
                self.long_pos = max([self.long_pos-quantity, 0])
            else:
                if quantity >= self.short_pos:
                    self.short_pos = 0
                    self.update(quantity - self.long_pos, OrderSide.BUY)
                self.short_pos = max([self.short_pos-quantity, 0])
        self.pos = self.long_pos-self.short_pos


class OptionTradeBlotter:

    def __init__(self):
        columns = ['tick_time', 'order_id', 'option_code', 'side', 'price', 'quantity', 'commission']
        self.trade_blotters = DataFrame(columns=columns)

    def update_trade_blotters(self, order: Order):
        transactions = {
            'tick_time': order.update_time,
            'order_id': order.order_id,
            'option_code': order.option_code,
            'side': order.side,
            'price': order.order_price,
            'quantity': order.executed_qty,
            'commission': 0. if (order.side == OrderSide.SELL) and (order.executed_qty > 0) else 1.6
        }
        self.trade_blotters = self.trade_blotters.append(transactions, ignore_index=True)


class Exchange:

    def __init__(self):
        self.etf_tick = None
        self.option_tick = None
        self.option_chain = None
        self.static_option_data = None
        self.call_option_code = list()
        self.put_option_code = list()
        self.current_tick_time = None
        self.current_tick_data = None

        self.day = ""
        self.month = ""
        self.year = 244.
        self.end_date = ""
        self.etf_name = ""
        self.underlying = ""
        self.t_tick = "13:01:00"
        self.strikes = list()

        self.order_ids = unique(random.randint(1e7, 1e8, size=100000))
        self.order_id_idx = 0

        self.open_orders = list()
        self.filled_orders = list()
        self.canceled_orders = list()
        self.position = dict()
        self.option_trade_blotter = OptionTradeBlotter()

        self.match_threshold = list()

        self.call_volume = array([])
        self.put_volume = array([])

        self.exchange_event_engine = EventEngine()
        self.init_event_engine()

    def init_event_engine(self):
        self.exchange_event_engine.register(EventType.EVENT_PLACE_ORDER, self.on_place_order)
        self.exchange_event_engine.register(EventType.EVENT_CANCEL_ORDER, self.on_cancel_order)
        self.exchange_event_engine.start()

    def init_position(self):
        self.position = {self.static_option_data.at[idx, 'option_code']: SingleOptionPosition(
            strike=self.static_option_data.at[idx, 'strike_price'],
            option_type=self.static_option_data.at[idx, 'call_put']
        ) for idx in self.static_option_data.index}

    def load_tick_from_csv(self, strategy_params: StrategyParams):
        self.day = strategy_params.back_test_date
        self.t_tick = self.day + " " + self.t_tick
        self.month = strategy_params.month
        self.year = strategy_params.year
        self.etf_name = strategy_params.etf_name
        self.underlying = strategy_params.underlying

        path = os.path.join("./exchange/", 'data', self.etf_name, self.day, '')
        self.etf_tick = TickQuotation(name=self.underlying)
        self.etf_tick.read_tick_csv(path=path)
        self.option_tick = ETFOptionTick(underlying=self.underlying, end_date=self.day)
        self.option_tick.read_tick_csv(path=path)

        self.option_chain = self.option_tick.option_chain
        self.static_option_data = self.option_chain.full_with_month(self.month)
        self.strikes = self.option_chain.strikes_with_month(month=self.month)
        n = len(self.strikes)
        self.call_volume = zeros(shape=n)
        self.put_volume = zeros(shape=n)
        self.match_threshold = linspace(0.1, 0.2, n//2).tolist() + linspace(0.2, 0.1, n-n//2).tolist()
        self.call_option_code = self.option_chain.option_code(month=self.month, option_type='认购')
        self.put_option_code = self.option_chain.option_code(month=self.month, option_type='认沽')
        self.init_position()

    def generate_tick_data(self, tick_time):
        etf_d = self.etf_tick.last_data_dict(tick_time=tick_time)
        spot = etf_d['last_price']
        spot_ask = etf_d['ask']
        spot_bid = etf_d['bid']
        option = self.option_tick.last_data(tick_time=tick_time)
        expire = option.option_chain.expires_with_month(month=self.month)[0]
        call = option.get_smile_tick_by_month(month=self.month, option_type='认购')
        put = option.get_smile_tick_by_month(month=self.month, option_type='认沽')
        tick_data = TickData(tick_time, spot, spot_ask, spot_bid, expire, call, put)
        return tick_data

    def generate_order_id(self):
        order_id = self.order_ids[self.order_id_idx]
        self.order_id_idx += 1
        self.order_id_idx %= len(self.order_ids)
        return str(order_id)

    def update_position(self, open_order: Order):
        option_code = open_order.option_code
        quantity = open_order.executed_qty
        side = open_order.side
        self.position[option_code].update(quantity, side)

    def match_orders(self):
        remove_list = list()
        call = self.current_tick_data.call
        put = self.current_tick_data.put
        diff_call_volume = call['volume']-self.call_volume
        diff_put_volume = put['volume']-self.put_volume
        self.call_volume = call['volume']
        self.put_volume = put['volume']
        random_matched_order = list()
        random.shuffle(self.open_orders)
        for open_order in self.open_orders:
            if open_order.order_status != OrderStatus.NEW:
                remove_list.append(open_order)
                continue
            option_code = open_order.option_code
            strike = self.static_option_data.loc[
                self.static_option_data['option_code'] == option_code, 'strike_price'].values[0]
            match_threhold = self.match_threshold[self.strikes.index(strike)]
            ask = bid = diff_volume = nan
            if option_code in self.call_option_code:
                ask = call.loc[call['option_code'] == option_code, 'ask'].values[0]
                bid = call.loc[call['option_code'] == option_code, 'bid'].values[0]
                diff_volume = diff_call_volume[strike]
            elif option_code in self.put_option_code:
                ask = put.loc[put['option_code'] == option_code, 'ask'].values[0]
                bid = put.loc[put['option_code'] == option_code, 'bid'].values[0]
                diff_volume = diff_put_volume[strike]
            side = open_order.side
            order_price = open_order.order_price
            order_time = open_order.order_time
            open_order.update_time = self.current_tick_time
            filled = False
            rand = random.random()
            if side == OrderSide.BUY:
                if ask <= order_price:
                    open_order.order_price = ask if order_time == self.current_tick_time else open_order.order_price
                    open_order.executed_qty = open_order.orig_qty
                    filled = True
                elif (rand <= match_threhold) and (option_code not in random_matched_order) \
                        and (order_price >= bid) and (diff_volume > 0):
                    random_matched_order.append(option_code)
                    open_order.executed_qty = int(40*match_threhold)
                    filled = True
            elif side == OrderSide.SELL:
                if bid >= order_price:
                    open_order.order_price = bid if order_time == self.current_tick_time else open_order.order_price
                    open_order.executed_qty = open_order.orig_qty
                    filled = True
                elif (rand < match_threhold) and (option_code not in random_matched_order) \
                        and (order_price <= ask) and (diff_volume > 0):
                    random_matched_order.append(option_code)
                    open_order.executed_qty = int(40*match_threhold)
                    filled = True
            if filled:
                open_order.order_status = OrderStatus.FILLED
                self.filled_orders.append(open_order)
                self.update_position(open_order)
                self.option_trade_blotter.update_trade_blotters(open_order)
        for open_order in remove_list:
            self.open_orders.remove(open_order)

    def _run(self, event: Event):
        ticks = self.etf_tick.tick
        ticks = ticks[ticks >= to_datetime(self.t_tick)]
        main_engine = event.data
        for tick_time in ticks:
            # push tick data
            self.current_tick_time = tick_time
            self.current_tick_data = self.generate_tick_data(tick_time)
            self.match_orders()
            main_engine.strategy.on_ticker(self.current_tick_data)
            # check orders
            self.match_orders()

    def on_place_order(self, event: Event):
        order: PlaceOrder = event.data
        order_id = self.generate_order_id()
        open_order = Order(order_id, order.order_time, order.order_time, order.option_code,
                           order.side, OrderStatus.NEW, order.order_type, order.price, order.qty, 0)
        if order.qty:
            self.open_orders.append(open_order)

    def on_cancel_order(self, event: Event):
        order_id = event.data
        for open_order in self.open_orders:
            if open_order.order_id == order_id:
                if open_order.order_status == OrderStatus.NEW:
                    open_order.order_status = OrderStatus.CANCELED
                    self.canceled_orders.append(open_order)
                break

    def cancel_all_orders(self):
        self.open_orders = list()
        self.canceled_orders += self.open_orders

    def get_option_quotes_via_code(self, code: str, price_type: str):
        if code in self.call_option_code:
            call = self.current_tick_data.call
            if price_type == 'bid':
                call_bid = call['bid'].values
                return call_bid[self.call_option_code.index(code)]
            elif price_type == 'ask':
                call_ask = call['ask'].values
                return call_ask[self.call_option_code.index(code)]
            else:
                print('Invalid Price Type!!!')
        elif code in self.put_option_code:
            put = self.current_tick_data.put
            if price_type == 'bid':
                put_bid = put['bid'].values
                return put_bid[self.put_option_code.index(code)]
            elif price_type == 'ask':
                put_ask = put['ask'].values
                return put_ask[self.put_option_code.index(code)]
            else:
                print('Invalid Price Type!!!')
        else:
            print('Option code not Found')
