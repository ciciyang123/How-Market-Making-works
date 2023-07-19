from enum import Enum
from typing import Any
from dataclasses import dataclass


class EventType(Enum):
    # Client
    EVENT_TICKER = "EVENT_TICKER"
    EVENT_POS = "EVENT_POS"
    EVENT_OPEN_ORDERS = "EVENT_OPEN_ORDERS"
    EVENT_TIMER = "EVENT_TIMER"

    # Exchange
    EVENT_SUBSCRIBE_TICKER = "SUBSCRIBE_TICKER"
    EVENT_PLACE_ORDER = "PLACE_ORDER"
    EVENT_CANCEL_ORDER = "CANCEL_ORDER"


@dataclass
class Event:
    event_type: EventType
    data: Any = None
