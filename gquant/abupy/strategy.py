class SellStrategy_SAR(abupy.AbuFactorSellBase):
    """SAR抛物线止损方案。当sar值大于或等于指定的对比值时卖出。"""

    def _init_self(self, **kwargs):
        """

        Args:
            sar (str): sar数据的列名。默认为`sar`。
            price (str): 当前数据的列名。默认为`close`。
        """
        self.sar = kwargs.pop('sar', 'sar')
        self.price = kwargs.pop('price', 'close')
        self.factor_name = '{}:price={},sar={}'.format(self.__class__.__name__,
                                                       self.price, self.sar)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [abupy.ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        if today[self.sar] >= today[self.price]:
            for order in orders:
                self.sell_tomorrow(order)


class SellStrategy_BBands(abupy.AbuFactorSellBase):
    """布林带卖出策略。当收盘价在中线以下时卖出"""

    def _init_self(self, **kwargs):
        self.sell_type_extra = self.__class__.__name__

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [abupy.ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        if today.close < today.bbmid:
            for order in orders:
                self.sell_tomorrow(order)


class BuyStrategy_BBands(abupy.AbuFactorBuyTD, abupy.BuyCallMixin):
    """布林带买入策略。当中线向上+开口变大+收盘价在上线以上时买入"""

    def _init_self(self, **kwargs):
        self.factor_name = '{}'.format(self.__class__.__name__)
        self.atr = kwargs['atr']

    def fit_day(self, today):
        t = today
        y = self.yesterday
        by = self.bf_yesterday
        b1 = t.bbmid > y.bbmid > by.bbmid  # 中线向上
        b2 = t.bbup - t.bblow > y.bbup - y.bblow > by.bbup - by.bblow  # 开口变大
        b3 = t.close > t.bbmid  # 收盘价在上线以上
        if b1 and b2 and b3:
            return self.buy_tomorrow()
        return None


class BuyStrategy_TDTP(abupy.AbuFactorBuyXD, abupy.BuyCallMixin):
    """通道突破买入方案。默认使用收盘价作为价格比较列。
    当价格超过`xd`天最高价时，第二日买入。参考:py:class:`abupy.FactorBuyBu.ABuFactorBuyBreak.AbuFactorBuyXD`。（区别在于计算xd_kl时去除当日）"""

    def read_fit_day(self, today):
        """覆盖base函数完成过滤统计周期内前xd天以及为fit_day中切片周期金融时间序列数据

        Args:
            today: 当前驱动的交易日金融时间序列数据

        Returns:
            生成的交易订单AbuOrder对象
        """
        if self.skip_days > 0:
            self.skip_days -= 1
            return None

        # 今天这个交易日在整个金融时间序列的序号
        self.today_ind = int(today.key)
        # 回测中默认忽略最后一个交易日
        if self.today_ind >= self.kl_pd.shape[0] - 1:
            return None

        # 忽略不符合买入的天（统计周期内前xd天）
        # 去除当日
        if self.today_ind < self.xd:
            return None

        # 完成为fit_day中切片周期金融时间序列数据
        # 去除当日
        self.xd_kl = self.kl_pd[self.today_ind - self.xd:self.today_ind]

        return self.fit_day(today)

    def _init_self(self, **kwargs):
        """

        Args:
            xd (int): 突破周期参数。比如20，30，40天...突破。
            price (str): 当前数据的列名。默认为`close`。
        """
        super()._init_self(**kwargs)
        self.price = kwargs.pop('price', 'close')
        # self.xd->突破周期参数 xd， 比如20，30，40天...突破
        self.factor_name = '{}:{}:xd={}'.format(self.__class__.__name__,
                                                self.price, self.xd)

    def fit_day(self, today):
        if today[self.price] > self.xd_kl[self.price].max():
            return self.buy_tomorrow()
        return None


class SellStrategy_TDTP(abupy.AbuFactorSellXD):
    """通道突破卖出方案。默认使用收盘价作为价格比较列。
    当价格低于`xd`天最低价时，第二日卖出。参考:py:class:`abupy.FactorSellBu.ABuFactorSellBreak.AbuFactorSellXDBK`。"""

    def _init_self(self, **kwargs):
        """

        Args:
            xd (int): 突破周期参数。比如20，30，40天...突破。
            price (str): 当前数据的列名。默认为`close`。
        """
        super()._init_self(**kwargs)
        self.price = kwargs.pop('price', 'close')
        # self.xd->突破周期参数 xd， 比如20，30，40天...突破
        self.factor_name = '{}:{}:xd={}'.format(self.__class__.__name__,
                                                self.price, self.xd)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [abupy.ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        if today[self.price] < self.xd_kl[self.price].min():
            for order in orders:
                self.sell_tomorrow(order)


class SellStrategy_NDay(abupy.AbuFactorSellBase):
    """N日卖出方案。持有N日后，如果没有盈利，则卖出。参考`:py:class:abupy.FactorSellBu.ABuFactorSellNDay`"""

    def _init_self(self, **kwargs):
        """

        Args:
            sell_n (int): 代表买入后持有的天数，默认1天。
            price (str): 当前数据的列名。默认为`close`。
        """
        self.sell_n = kwargs.pop('sell_n', 1)
        self.price = kwargs.pop('price', 'close')
        self.sell_type_extra = '{}:sell_n={}'.format(self.__class__.__name__,
                                                     self.sell_n)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [abupy.ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        for order in orders:
            # 将单子的持有天数进行增加
            order.keep_days += 1
            """
                today.close - order.buy_price：截止今天相比买入时的收益，
                order.expect_direction：买单的方向，收益＊方向＝实际收益
            """
            profit = (today.close - order.buy_price) * order.expect_direction
            if order.keep_days >= self.sell_n and profit <= 0:
                # 超过self.sell_n，并且当前价格<=买入价格，即卖出
                self.sell_tomorrow(order)


class SellStrategy_ATR(abupy.AbuFactorSellBase):
    """n倍atr(止盈止损)。派生自:py:class`abupy.FactorSellBu.ABuFactorAtrNStop`。因为原本无法指定atr数据来源。

    Args:
        atr (str): atr数据的列名。默认为`atr`。
        stop_loss_n (float): 止损的atr倍数。
        stop_win_n (float): 止盈的atr倍数。
    """

    def _init_self(self, **kwargs):
        if 'stop_loss_n' in kwargs:
            # 设置止损的atr倍数
            self.stop_loss_n = kwargs['stop_loss_n']
            # 在输出生成的orders_pd中及可视化等等显示的名字
            self.sell_type_extra_loss = '{}:stop_loss={}'.format(
                self.__class__.__name__, self.stop_loss_n)

        if 'stop_win_n' in kwargs:
            # 设置止盈的atr倍数
            self.stop_win_n = kwargs['stop_win_n']
            # 在输出生成的orders_pd中及可视化等等显示的名字
            self.sell_type_extra_win = '{}:stop_win={}'.format(
                self.__class__.__name__, self.stop_win_n)

        self.atr = kwargs['atr']
        self.factor_name = '{}:atr={}'.format(self.__class__.__name__,
                                              self.atr)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [abupy.ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        """
        止盈event：截止今天相比买入时的收益 * 买入时的期望方向 > n倍atr
        止损event：截止今天相比买入时的收益 * 买入时的期望方向 < -n倍atr
        :param today: 当前驱动的交易日金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        :return:
        """

        for order in orders:
            """
                today.close - order.buy_price：截止今天相比买入时的收益，
                order.expect_direction：买单的方向，收益＊方向＝实际收益
            """
            profit = (today.close - order.buy_price) * order.expect_direction
            # 使用购买日当前的ATR
            stop_base = self.kl_pd[self.kl_pd['date'] == order.buy_date]
            stop_base = stop_base[self.atr][0]
            if hasattr(
                    self, 'stop_win_n'
            ) and profit > 0 and profit > self.stop_win_n * stop_base:
                # 满足止盈条件卖出股票, 即收益(profit) > n倍atr
                self.sell_type_extra = self.sell_type_extra_win
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_tomorrow(order)

            if hasattr(
                    self, 'stop_loss_n'
            ) and profit < 0 and profit < -self.stop_loss_n * stop_base:
                # 满足止损条件卖出股票, 即收益(profit) < -n倍atr
                self.sell_type_extra = self.sell_type_extra_loss
                order.fit_sell_order(self.today_ind, self)
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_tomorrow(order)
