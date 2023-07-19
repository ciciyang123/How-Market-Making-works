from enum import Enum
from gateway.optionClient import SSEOptionClient
from gateway.mktData import MktData
from gateway.quotes import Quotes
from strategies.strategyParams import StrategyParams
from utils.event.eventEngine import EventEngine


class MainEngineStatus(Enum):
    READY = "READY"
    START = "START"
    STOP = "STOP"


class MainEngine(object):

    def __init__(self, event_engine: EventEngine, option_client: SSEOptionClient):
        self.event_engine = event_engine
        self.strategy = None
        self.status = MainEngineStatus.READY
        self.mkt_data = MktData(self, option_client)
        self.quotes = Quotes(event_engine, option_client)

    def start(self):
        self.status = MainEngineStatus.START
        self.event_engine.start()
        self.run_jobs()

    def stop(self):
        self.status = MainEngineStatus.STOP
        self.event_engine.stop()

    def add_strategy(self, strategy_class, strategy_params: StrategyParams):
        self.strategy = strategy_class(self, strategy_params)

    def run_jobs(self):
        self.mkt_data.subscribe(self.strategy.strategy_params)

