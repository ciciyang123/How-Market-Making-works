from enum import Enum
from strategies.strategyParams import StrategyParams
from utils.mainEngine import MainEngine
from utils.event.event import EventType, Event


class StrategyStatus(Enum):
    OBSERVING = "OBSERVING"
    QUOTING = "QUOTING"
    HEDGING = "HEDGING"


class BaseStrategy(object):

    def __init__(self, main_engine: MainEngine, strategy_params: StrategyParams):
        self.main_engine = main_engine
        self.strategy_params = strategy_params

        self.main_engine.event_engine.register(EventType.EVENT_POS, self.on_pos)
        self.main_engine.event_engine.register(EventType.EVENT_OPEN_ORDERS, self.on_open_orders)
        self.main_engine.event_engine.register(EventType.EVENT_TICKER, self.on_ticker)
        self.main_engine.event_engine.register(EventType.EVENT_TIMER, self.on_timer)

    def on_pos(self, event: Event):
        pass

    def on_open_orders(self, event: Event):
        pass

    def on_ticker(self, event: Event):
        pass

    def on_timer(self, event: Event):
        pass
