from exchange.optionExchange import Exchange
from gateway.dataFormat import *
from utils.event.event import *


class SSEOptionClient(object):

    def __init__(self):
        self.exchange = Exchange()

    @property
    def _server_time(self):
        return self.exchange.current_tick_time

    def place_order(self, option_code: str, side: OrderSide, price: float,
                    quantity: float, order_type: OrderType = OrderType.LIMIT):
        order_time = self._server_time
        order = PlaceOrder(order_time, option_code, side, price, quantity, order_type)
        event = Event(EventType.EVENT_PLACE_ORDER, data=order)
        self.exchange.on_place_order(event)

    def cancel_order(self, order_id=None):
        event = Event(EventType.EVENT_CANCEL_ORDER, data=order_id)
        self.exchange.on_cancel_order(event)

    def cancel_all_orders(self):
        self.exchange.cancel_all_orders()

    def get_latest_tick_data(self):
        return self.exchange.current_tick_data

    def get_open_orders(self):
        return self.exchange.open_orders

    def get_filled_orders(self):
        return self.exchange.filled_orders

    def get_position(self):
        return self.exchange.position

    def get_trade_blotters(self):
        return self.exchange.option_trade_blotter.trade_blotters

    def get_option_chain(self):
        return self.exchange.option_chain.option_chain

    def get_option_quotes_via_code(self, code: str, price_type: str):
        return self.exchange.get_option_quotes_via_code(code, price_type)

