class StrategyParams:

    def __init__(self,
                 strategy_name: str = "",
                 etf_name: str = "",
                 underlying: str = "",
                 back_test_date: str = "",
                 month: str = "",
                 r: float = 0,
                 q: float = 0,
                 unit: int = 0,
                 call_r: float = 0,
                 put_r: float = 0,
                 year: float = 244,
                 max_iv_gap: float = 0.01, #is this the iv diff between the largest and smallest
                 otm_iv_weight: float = 0.07,
                 ):
        self.strategy_name = strategy_name
        self.etf_name = etf_name
        self.underlying = underlying
        self.back_test_date = back_test_date
        self.month = month
        self.r = r
        self.q = q
        self.unit = unit
        self.call_r = call_r
        self.put_r = put_r
        self.year = year
        self.max_iv_gap = max_iv_gap
        self.otm_iv_weight = otm_iv_weight

    def set_call_r(self, call_r: float):
        self.call_r = call_r

    def set_put_r(self, put_r: float):
        self.put_r = put_r

    def set_max_iv_gap(self, max_iv_gap: float):
        self.max_iv_gap = max_iv_gap

    def set_otm_iv_weight(self, otm_iv_weight: float):
        self.otm_iv_weight = otm_iv_weight

    def get_strategy_name(self):
        return self.strategy_name

    def get_underlying(self):
        return self.underlying
