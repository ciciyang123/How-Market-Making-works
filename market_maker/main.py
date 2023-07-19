
from gateway.optionClient import SSEOptionClient
from utils.mainEngine import MainEngine
from utils.event.eventEngine import EventEngine
from strategies.marketMakerStrategy import MarketMakerStrategy
from strategies.strategyParams import StrategyParams

if __name__ == "__main__":
    event_engine = EventEngine()
    option_client = SSEOptionClient()
    main_engine = MainEngine(event_engine=event_engine, option_client=option_client)
    strategy_params = StrategyParams(
        strategy_name="Market Maker",
        etf_name="300ETF_SH",
        underlying="510300.SH",
        back_test_date="2021-01-05",
        month="2102",
        r=0.0225,
        q=0.0225,
        unit=10000,
        call_r=0.01,
        put_r=0.025,
        year=244.0,
    )
    main_engine.add_strategy(MarketMakerStrategy, strategy_params)
    main_engine.start()


