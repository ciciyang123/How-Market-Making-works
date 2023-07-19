from numpy import array
from pandas import Series


class ActionSpace:

    def __init__(self, unit=1e-3):
        self.unit = unit
        self.normal = dict()
        self.buy_vol = dict()
        self.sell_vol = dict()
        self.buy_put_sell_call = dict()
        self.buy_call_sell_put = dict()
        self.avert = dict()
        self.hedge = "hedge"
        self.last_action = -1

    def set_variables(self, n_strikes: int, atm_index: int, unit=None):
        if unit:
            self.unit = unit
        self.normal = {'bid': [-2 * self.unit] * n_strikes, 'ask': [2 * self.unit] * n_strikes}
        self.buy_vol = {'bid': [-1 * self.unit] * n_strikes, 'ask': [3 * self.unit] * n_strikes}
        self.sell_vol = {'bid': [-3 * self.unit] * n_strikes, 'ask': [1 * self.unit] * n_strikes}
        self.buy_put_sell_call = {'bid': [-1 * self.unit] * atm_index + [-3 * self.unit] * (n_strikes - atm_index),
                                  'ask': [3 * self.unit] * atm_index + [1 * self.unit] * (n_strikes - atm_index)}
        self.buy_call_sell_put = {'bid': [-5 * self.unit] * atm_index + [-1 * self.unit] * (n_strikes - atm_index),
                                  'ask': [1 * self.unit] * atm_index + [3 * self.unit] * (n_strikes - atm_index)}
        self.avert = {'bid': [-3 * self.unit] * n_strikes, 'ask': [3 * self.unit] * n_strikes}

    def get_action(self, action):
        if action == 0:
            return self.normal
        elif action == 1:
            return self.buy_vol
        elif action == 2:
            return self.sell_vol
        elif action == 3:
            return self.buy_put_sell_call
        elif action == 4:
            return self.buy_call_sell_put
        elif action == 5:
            return self.avert
        elif action == 6:
            return self.hedge
        else:
            raise Exception("action not found")

    def update_action(self, action):
        self.last_action = action

class RewardFunction:

    def __init__(self):
        self.reward = 0.
        self.last_pnl = 0.
        self.cum_reward = 0.

    def update_reward(self, pnl: float):
        self.reward = pnl - self.last_pnl
        self.last_pnl = pnl
        self.cum_reward += self.reward
