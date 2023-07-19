from exchange.TickQuotation.TickQuotation import TickQuotation, Tick_String
from exchange.TickQuotation.ETFOptionChain import ETFOptionChain
from pandas import DataFrame
from numpy import array, float64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class ETFOptionCurrentTick:

    def __init__(self, option_chain: ETFOptionChain):  # underlying: array, strike: array, expire: array):
        
        self.option_chain = option_chain
        self.current_tick_str = ['strike', 'expire', 'call_put', 'option_code']
        tick_frame = self.option_chain.strikes_expires()
        self.underlying = option_chain.underlying
        index = range(0, len(tick_frame))
        self.tick = DataFrame(index=index, columns=self.current_tick_str+Tick_String, dtype=float64)
        self.tick[self.current_tick_str] = tick_frame[self.current_tick_str]
        self.current_tick_str += Tick_String
        n = len(tick_frame)
        self.tick.loc[n, ['strike', 'expire', 'call_put', 'option_code']] = (-1, -1, None, None)
        self.tick = self.tick.set_index(keys=['strike', 'expire', 'call_put'])

    def size(self):
        return len(self.tick) - 1

    # def get_tick_by_option(self, exchange: str):
    #     chain = self.option_chain.option_chain
    #     index = chain[chain['option_code'] == exchange].index
    #     expire = chain.loc[index, 'expiredate']
    #     strike = chain.loc[index, 'strike']

    def get_strike_by_month(self, month: str):
        return self.option_chain.strikes_with_month(month=month)

    def get_expire_by_month(self, month: str):
        return self.option_chain.expires_with_month(month=month)[0]

    def get_smile_tick_by_month(self, month: str, option_type: str):
        expire = self.get_expire_by_month(month=month)
        x = self.tick.xs(expire, level='expire')
        x = x.xs(option_type, level='call_put')
        return x

    def set_value(self, keys, ticks: []):
        self.tick.loc[keys, Tick_String] = ticks


class ETFOptionTick:

    def __init__(self, underlying: str, end_date: str):
        self.path = ""
        self.option_chain = ETFOptionChain(
            underlying=underlying,
            end_date=end_date
        )
        self.option_tick = list()
        chain = self.option_chain
        self.current_tick = ETFOptionCurrentTick(option_chain=chain)

    def __chain_size(self) -> int:
        return self.option_chain.size()

    def load_tick(self, begin: str, end: str):
        for i in range(0, self.__chain_size()):
            option = self.option_chain.loc_option(index=i)
            tick = TickQuotation(option)
            tick.load_tick(begin=begin, end=end)
            self.option_tick.append(tick)

    def reserve(self, month: array):
        self.option_chain.reserve(month=month)

    #                   #
    #   tick operation  #
    #                   #
    def last_tick(self, tick_time: datetime) -> array:
        tick = list()
        for i in range(0, self.__chain_size()):
            tick.append(self.option_tick[i].last_tick(tick_time=tick_time))
        return tick

    #                   #
    #   data operation  #
    #                   #
    def last_data(self, tick_time: datetime):
        
        keys = []
        ticks = []
        for i in range(0, self.__chain_size()):
            strike = self.option_chain.loc_strike(index=i)
            expire = self.option_chain.loc_expire(index=i)
            option_type = self.option_chain.loc_type(index=i)
            tick = self.option_tick[i].last_data(tick_time=tick_time)
            keys.append((strike, expire, option_type))
            ticks.append(tick)
        self.current_tick.set_value(keys=keys, ticks=ticks)
        return self.current_tick

    #                   #
    #    persistence    #
    #                   #
    def to_tick_csv(self, path: str):
        
        for option_tick in self.option_tick:
            option_tick.to_tick_csv(path=path)

    def read_single_tick_csv(self, name: str):
        option_tick = TickQuotation(name=name)
        option_tick.read_tick_csv(path=self.path)
        return option_tick

    def read_tick_csv(self, path: str):
        
        self.option_tick.clear()
        self.path = path
        options = self.option_chain.options()
        arg_list = [name for name in options]
        with ThreadPoolExecutor() as pool:
            results = pool.map(self.read_single_tick_csv, arg_list)
        for result in results:
            self.option_tick.append(result)
