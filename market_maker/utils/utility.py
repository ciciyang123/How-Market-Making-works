from datetime import timedelta
from pandas import to_datetime


def get_trading_seconds(tick_time):
    date = str(tick_time)[:10]
    morning_opening = to_datetime(f"{date} 09:30:00.000")
    afternoon_opening = to_datetime(f"{date} 13:00:00.000")
    if tick_time <= morning_opening:
        tick_time = morning_opening
    elif tick_time >= afternoon_opening:
        tick_time -= timedelta(hours=1, minutes=30)
    trading_seconds = (tick_time - morning_opening).total_seconds()
    return trading_seconds
