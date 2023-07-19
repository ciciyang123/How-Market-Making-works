from dataclasses import dataclass
from pandas import Timestamp, DataFrame
from enum import Enum


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TickData:
    tick_time: Timestamp
    spot: float
    spot_ask: float
    spot_bid: float
    expire: float
    call: DataFrame
    put: DataFrame


@dataclass
class PlaceOrder:
    order_time: Timestamp
    option_code: str
    side: OrderSide
    price: float
    qty: float
    order_type: OrderType


@dataclass
class Order:
    order_id: str
    order_time: Timestamp
    update_time: Timestamp
    option_code: str
    side: OrderSide
    order_status: OrderStatus
    order_type: OrderType
    order_price: float
    orig_qty: float
    executed_qty: float

    def __repr__(self):
        return self.order_id
