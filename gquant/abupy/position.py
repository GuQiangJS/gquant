class Position_Atr(abupy.AbuPositionBase):
    """atr仓位管理。默认使用10日atr值进行计算。与:py:class`abupy.BetaBu.ABuAtrPosition`不同。
        1. 因为原本无法指定atr数据来源。
        2. abupy中的没有看懂 :(

    Attributes:
        atr (str): 默认使用10日atr值进行计算。
        pos_base (float): 最大亏损占比。（例如占初始资金1%时，传入0.01)。默认为0.01。
    """
    def _init_self(self, **kwargs):
        super()._init_self(**kwargs)
        self.atr = kwargs.pop('atr', 'atr10')
        self.g_atr_pos_base = kwargs.pop('pos_base', 0.01)

    def fit_position(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        计算：（常数价格 ／ 买入价格）＊ 当天交易日atr21
        :param factor_object: ABuFactorBuyBases实例对象
        :return:买入多少个单位（股，手，顿，合约）
        """
        if self.atr not in self.kl_pd_buy:
            raise ValueError()
        # std_atr = (abupy.AbuAtrPosition.s_atr_base_price /
        #            self.bp) * self.kl_pd_buy[self.atr]
        # """
        #     对atr 进行限制 避免由于股价波动过小，导致
        #     atr小，产生大量买单，实际上针对这种波动异常（过小，过大）的股票
        #     需要有其它的筛选过滤策略, 选股的时候取0.5，这样最大取两倍g_atr_pos_base
        # """
        # atr_wv = abupy.AbuAtrPosition.s_std_atr_threshold if std_atr < abupy.AbuAtrPosition.s_std_atr_threshold else std_atr
        # # 计算出仓位比例
        # atr_pos = self.g_atr_pos_base / atr_wv
        # # 最大仓位限制
        # atr_pos = abupy.ABuPositionBase.g_pos_max if atr_pos > abupy.ABuPositionBase.g_pos_max else atr_pos
        # # 结果是买入多少个单位（股，手，顿，合约）
        # return self.read_cash * atr_pos / self.bp * self.deposit_rate
        return self.read_cash * self.g_atr_pos_base / self.bp / self.kl_pd_buy[
            self.atr]


class Position_AllIn(abupy.AbuPositionBase):
    """全仓"""
    def _init_self(self, **kwargs):
        """子类仓位管理针对可扩展参数的初始化"""
        pass

    def fit_position(self, factor_object):
        return self.read_cash / self.bp
