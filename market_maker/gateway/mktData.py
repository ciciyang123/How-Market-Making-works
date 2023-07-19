from gateway.optionClient import SSEOptionClient
from strategies.strategyParams import StrategyParams
from utils.event.event import *


class MktData:

    def __init__(self, owner, option_client: SSEOptionClient):
        self.option_client = option_client
        self.owner = owner

    def get_latest_tick_data(self):
        return self.option_client.get_latest_tick_data()

    def get_option_quotes_via_code(self, code: str, price_type: str):
        return self.option_client.get_option_quotes_via_code(code, price_type)

    def get_option_chain(self):
        return self.option_client.get_option_chain()

    def subscribe(self, strategy_params: StrategyParams):
        self.option_client.exchange.load_tick_from_csv(strategy_params)
        event = Event(EventType.EVENT_SUBSCRIBE_TICKER, data=self.owner)
        self.option_client.exchange._run(event)
        # self.option_client.exchange.exchange_event_engine.put(event)
