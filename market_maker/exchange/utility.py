import time
from numpy import random


def sleep_random_time(max_sleep_seconds: float = 3.):
    time.sleep(1 + (max_sleep_seconds - 1) * random.rand())

