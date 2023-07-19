from gateway.optionClient import *
from utils.event.event import EventType, Event
from utils.event.eventEngine import EventEngine


class Quotes:

    def __init__(self, event_engine: EventEngine, option_client: SSEOptionClient):
        self.option_client = option_client
        self.event_engine = event_engine

    def buy_open(self, option_code: str, price: float, quantity: float, order_type: OrderType = OrderType.LIMIT):
        self.option_client.place_order(option_code, OrderSide.BUY, price, quantity, order_type)

    def buy_close(self, option_code: str, price: float, quantity: float, order_type: OrderType = OrderType.LIMIT):
        self.option_client.place_order(option_code, OrderSide.SELL, price, quantity, order_type)

    def sell_open(self, option_code: str, price: float, quantity: float, order_type: OrderType = OrderType.LIMIT):
        self.option_client.place_order(option_code, OrderSide.SELL, price, quantity, order_type)

    def sell_close(self, option_code: str, price: float, quantity: float, order_type: OrderType = OrderType.LIMIT):
        self.option_client.place_order(option_code, OrderSide.BUY, price, quantity, order_type)

    def cancel_order(self, order_id: int):
        self.option_client.cancel_order(order_id)

    def cancel_all_orders(self):
        self.option_client.cancel_all_orders()

    def get_open_orders(self):
        return self.option_client.get_open_orders()

    def get_position(self):
        return self.option_client.get_position()

    def get_trade_blotters(self):
        return self.option_client.get_trade_blotters()

