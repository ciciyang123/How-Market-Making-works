# from WindPy import *
from numpy import array, nan
from enum import Enum
from datetime import datetime
from pandas import DataFrame, read_csv, to_datetime
from abc import abstractmethod

Tick_String = ['last_price', 'ask', 'bid', 'ask_vol', 'bid_vol', 'volume', 'open_interest', 'tick']


class QEState(Enum):
    NullData = -1
    LoadQuotationError = -2
    LoadHistoryFileNull = -3
    NoAsset = -4


class QuotationException(Exception):
    def __init__(self, state: QEState):
        self.state = state


class TickQuotation:

    # w.start()
    def __init__(self, name):
        # tick detail
        self.quotation_str = "last,ask1,asize1,bid1,bsize1,oi,volume"
        self.tick_str = Tick_String
        self.csv_str = ['last', 'ask1', 'asize1', 'bid1', 'bsize1', 'oi', 'volume']
        self.name = name
        # tick time
        self.tick = list()
        self.null_tick_time = to_datetime(0)
        self.current_tick = -1
        # tick data
        self.last_price = self.ask_price = self.ask_volume = self.bid_price = \
            self.bid_volume = self.open_interest = self.volume = list()

    # def load_tick(self, begin: str, end: str):
    #     wd = w.wst(self.name, self.quotation_str, begin, end, "")
    #     if wd.ErrorCode != 0:
    #         raise QuotationException(QEState.LoadQuotationError)
    #     self.tick = array(wd.Times)
    #     if self.size() > 0:
    #         self.current_tick = 0
    #         data = wd.Data
    #         self.last_price = array(data[0])
    #         self.ask_price = array(data[1])
    #         self.ask_volume = array(data[2])
    #         self.bid_price = array(data[3])
    #         self.bid_volume = array(data[4])
    #         self.open_interest = array(data[5])
    #         self.volume = array(data[6])

    #                   #
    #   tick operation  #
    #                   #
    def size(self) -> int:
        return len(self.tick)

    def null(self) -> bool:
        if self.size() <= 0:
            return True
        else:
            return False

    def end(self) -> bool:
        if self.null():
            return True
        if self.current_tick < self.size():
            return False
        else:
            return True

    def current_tick_time(self):
        return self.current_tick

    def next_tick(self) -> int:
        if self.null() is True:
            raise QuotationException(QEState.NullData)
        i = self.current_tick
        if self.end() is False:
            self.current_tick += 1
        return i

    def last_tick(self, tick_time: datetime):

        if self.size() <= 0:
            return -1
        if self.tick[0] > tick_time:
            return -1
        if self.size() - 1 == self.current_tick:
            return self.current_tick
        for i in range(self.current_tick, self.size()):
            if self.tick[i] > tick_time:
                self.current_tick = i
                return i - 1
        self.current_tick = self.size() - 1
        return self.current_tick

    #                   #
    #   data operation  #
    #                   #
    def append(self, tick: datetime, last_price: float, ask: float, bid: float,
               ask_vol: float, bid_vol: float, volume: float, oi: float):
        # ['last_price', 'ask', 'bid', 'ask_vol', 'bid_vol', 'volume', 'open_interest', 'tick']
        self.current_tick = 0
        self.tick.append(tick)
        self.last_price.append(last_price)
        self.ask_price.append(ask)
        self.bid_price.append(bid)
        self.ask_volume.append(ask_vol)
        self.bid_volume.append(bid_vol)
        self.volume.append(volume)
        self.open_interest.append(oi)

    def next_data(self):
        i = self.next_tick()
        t = self.tick[i]
        price = self.last_price[i]
        volume = self.volume[i]
        ask = self.ask_price[i]
        ask_vol = self.ask_volume[i]
        bid = self.bid_price[i]
        bid_vol = self.bid_volume[i]
        oi = self.open_interest[i]
        return price, ask, bid, ask_vol, bid_vol, volume, oi, t

    def last_data(self, tick_time: datetime):

        i = self.last_tick(tick_time=tick_time)

        if i == -1:
            t = self.null_tick_time
            price = volume = ask = ask_vol = bid = bid_vol = oi = nan
        else:
            t = self.tick[i]
            price = self.last_price[i]
            volume = self.volume[i]
            ask = self.ask_price[i]
            ask_vol = self.ask_volume[i]
            bid = self.bid_price[i]
            bid_vol = self.bid_volume[i]
            oi = self.open_interest[i]
        return price, ask, bid, ask_vol, bid_vol, volume, oi, t

    def last_data_dict(self, tick_time: datetime):
        d = self.last_data(tick_time=tick_time)
        last_data = {tick_str: d[i] for i, tick_str in enumerate(self.tick_str)}
        return last_data

    #                   #
    #    persistence    #
    #                   #
    def clear(self):
        self.current_tick = -1
        self.last_price = array([])
        self.ask_price = array([])
        self.ask_volume = array([])
        self.bid_price = array([])
        self.bid_volume = array([])
        self.open_interest = array([])
        self.volume = array([])

    def to_tick_csv(self, path: str):
        df = DataFrame(index=self.tick, columns=self.csv_str)
        df['last'] = self.last_price
        df['ask1'] = self.ask_price
        df['asize1'] = self.ask_volume
        df['bid1'] = self.bid_price
        df['bsize1'] = self.bid_volume
        df['oi'] = self.open_interest
        df['volume'] = self.volume
        filename = path + self.name + ".csv"
        df.to_csv(filename)

    def read_tick_csv(self, path: str):

        self.clear()
        filename = path + self.name + ".csv"
        df = read_csv(filename, index_col=0, header=0)

        self.tick = to_datetime(df.index)
        self.last_price = df['last'].values
        self.ask_price = df['ask1'].values
        self.ask_volume = df['asize1'].values
        self.bid_price = df['bid1'].values
        self.bid_volume = df['bsize1'].values
        self.open_interest = df['oi'].values
        self.volume = df['volume'].values

        if len(self.tick) > 0:
            self.current_tick = 0
        else:
            raise QuotationException(state=QEState.LoadHistoryFileNull)

    #                       #
    #    abstract method    #
    #                       #
    @abstractmethod
    def synthetic_tick_update(self, tick: array):
        pass
