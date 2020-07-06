import abupy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Metrics(abupy.AbuMetricsBase):
    """对:py:class:`abupy.AbuMetricsBase`的扩展。"""

    def _metrics_extend_stats(self):
        self.act_sell = self.action_pd[self.action_pd.action.isin(['sell'])
                                       & self.action_pd.deal.isin([True])]
        if self.act_sell.empty:
            return
        self.act_sell['sell_amount'] = self.act_sell.apply(
            lambda order: order.Price * order.Cnt, axis=1)
        self.act_sell['buy_amount'] = self.act_sell.apply(
            lambda order: order.Price2 * order.Cnt, axis=1)

        self.act_sell['sell_commission'] = self.act_sell.apply(
            lambda order: abupy.TradeBu.ABuCommission.calc_commission_cn(
                order.Cnt, order.Price),
            axis=1)
        self.act_sell['buy_commission'] = self.act_sell.apply(
            lambda order: abupy.TradeBu.ABuCommission.calc_commission_cn(
                order.Cnt, order.Price),
            axis=1)

        self.act_sell[
            'profit'] = self.act_sell.sell_amount - self.act_sell.buy_amount - self.act_sell.sell_commission - self.act_sell.buy_commission
        # 盈利比例
        self.act_sell['profit_cg'] = self.act_sell['profit'] / (
            self.act_sell['Price2'] * self.act_sell['Cnt'])
        # 为了显示方便及明显
        self.act_sell['profit_cg_hunder'] = self.act_sell['profit_cg'] * 100

        # 盈利交易
        self.ret = self.act_sell[self.act_sell.profit > 0]
        # 亏损交易
        self.los = self.act_sell[self.act_sell.profit <= 0]
        # 盈利交易平均盈利额
        self.avg_ret = self.ret['profit'].mean()
        # 亏损交易平均亏损额
        self.avg_los = self.los['profit'].mean()
        # 盈亏总额
        self.sum_profit = self.ret['profit'].sum()
        # 每笔交易平均盈亏额
        self.avg_profit = self.act_sell.profit.mean()
        # R=平均利润/平均损失
        self.R = np.round(abs(self.avg_profit) / abs(self.avg_los), 2)

    def profit_series(self, **kwargs):
        return pd.Series(data=[self.sum_profit,  # 盈亏总额
                               # 最终价值
                               self.capital.capital_pd.iloc[-1]['capital_blance'],
                               len(self.act_sell),  # 交易次数
                               len(self.ret),  # 盈利次数
                               len(self.los),  # 亏损次数
                               len(self.ret) / len(self.act_sell),  # 盈利比率
                               self.avg_profit,  # 每笔交易平均盈亏额
                               self.avg_ret,  # 盈利交易平均盈利额
                               self.avg_los,  # 亏损交易平均亏损额
                               self.R,  # R
                               self.max_drawdown],  # 最大回撤
                         index=['盈亏总额', '最终价值', '交易次数', '盈利次数', '亏损次数', '盈利比率',
                                '每笔交易平均盈亏额', '盈利交易平均盈利额', '亏损交易平均亏损额', 'R', '最大回撤'],
                         **kwargs)


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
        self.factor_name = '{}:price={},sar={}'.format(
            self.__class__.__name__, self.price, self.sar)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [abupy.ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        if today[self.sar] >= today[self.price]:
            for order in orders:
                self.sell_tomorrow(order)


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
        self.factor_name = '{}:{}:xd={}'.format(
            self.__class__.__name__, self.price, self.xd)

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
        self.factor_name = '{}:{}:xd={}'.format(
            self.__class__.__name__, self.price, self.xd)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [abupy.ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        if today[self.price] < self.xd_kl[self.price].max():
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
        self.sell_type_extra = '{}:sell_n={}'.format(
            self.__class__.__name__, self.sell_n)

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
            stop_base = today[self.atr]
            stop_base = today.atr21 + today.atr14
            if hasattr(self, 'stop_win_n') and profit > 0 and profit > self.stop_win_n * stop_base:
                # 满足止盈条件卖出股票, 即收益(profit) > n倍atr
                self.sell_type_extra = self.sell_type_extra_win
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_tomorrow(order)

            if hasattr(self, 'stop_loss_n') and profit < 0 and profit < -self.stop_loss_n * stop_base:
                # 满足止损条件卖出股票, 即收益(profit) < -n倍atr
                self.sell_type_extra = self.sell_type_extra_loss
                order.fit_sell_order(self.today_ind, self)
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_tomorrow(order)


class Position_Atr(abupy.AbuPositionBase):
    """atr仓位管理。复制并改写自:py:class`abupy.BetaBu.ABuAtrPosition`。因为原本无法指定atr数据来源。"""

    def _init_self(self, **kwargs):
        super()._init_self(**kwargs)
        self.atr = kwargs.pop('atr', 'atr21')
        self.g_atr_pos_base = kwargs.pop('pos_base', 0.1)

    def fit_position(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        计算：（常数价格 ／ 买入价格）＊ 当天交易日atr21
        :param factor_object: ABuFactorBuyBases实例对象
        :return:买入多少个单位（股，手，顿，合约）
        """
        if self.atr not in self.kl_pd_buy:
            raise ValueError()
        std_atr = (abupy.AbuAtrPosition.s_atr_base_price /
                   self.bp) * self.kl_pd_buy[self.atr]

        """
            对atr 进行限制 避免由于股价波动过小，导致
            atr小，产生大量买单，实际上针对这种波动异常（过小，过大）的股票
            需要有其它的筛选过滤策略, 选股的时候取0.5，这样最大取两倍g_atr_pos_base
        """
        atr_wv = abupy.AbuAtrPosition.s_std_atr_threshold if std_atr < abupy.AbuAtrPosition.s_std_atr_threshold else std_atr
        # 计算出仓位比例
        atr_pos = self.g_atr_pos_base / atr_wv
        # 最大仓位限制
        atr_pos = abupy.ABuPositionBase.g_pos_max if atr_pos > abupy.ABuPositionBase.g_pos_max else atr_pos
        # 结果是买入多少个单位（股，手，顿，合约）
        return self.read_cash * atr_pos / self.bp * self.deposit_rate


class Position_AllIn(abupy.AbuPositionBase):
    """全仓"""

    def _init_self(self, **kwargs):
        """子类仓位管理针对可扩展参数的初始化"""
        pass

    def fit_position(self, factor_object):
        return self.read_cash / self.bp

class MetricsUtils():
    def _plot_dist(series, benchmark,**kwargs):
        ax = sns.distplot(series,**kwargs)
        if benchmark:
            ax.axvline(benchmark, 0, 1, color='r')
            ax.text(benchmark, 0, 'benchmark:{:.4f}'.format(benchmark))
        return ax

    def plot_all(metrics, **kwargs):
        """绘图

        Args:
            init_cash (float): 初始资金。用于在最终资金图中绘制初始资金竖直线。默认为None。
            start (str): 测试开始时间。用于绘制标题。
            end (str): 测试结束时间。用于绘制标题。
        """
        figsize = kwargs.pop('figsize', (15, 15))
        fig, axes = plt.subplots(3, 3, figsize=figsize)

        R = pd.Series([m.R for m in metrics if hasattr(m, 'R')]).dropna()
        _plot_dist(R, None, ax=axes[0, 0])
        axes[0, 0].axvline(x=R.mean(), color='#d62728',
                           label='R 均值:{:.2f}'.format(R.mean()))
        axes[0, 0].set_title('R值')
        axes[0, 0].legend()

        MD = pd.Series(
            [m.max_drawdown for m in metrics if hasattr(m, 'max_drawdown')])
        _plot_dist(MD, None, ax=axes[0, 1])
        axes[0, 1].axvline(x=np.mean(MD), color='#d62728',
                           label='最大回撤均值:{:.2%}'.format(np.mean(MD)))
        axes[0, 1].set_title('最大回撤')
        axes[0, 1].legend()

        AVG_WIN = [m.avg_ret for m in metrics if hasattr(
            m, 'avg_ret') and m.avg_ret is not np.NaN]
        _plot_dist(pd.Series(AVG_WIN), None, label='盈利交易平均盈利额均值:{:.2f}'.format(
            np.mean(AVG_WIN)), ax=axes[0, 2])
        axes[0, 2].axvline(x=np.mean(AVG_WIN), color='#d62728')
        AVG_LOS = [m.avg_los for m in metrics if hasattr(
            m, 'avg_los') and m.avg_los is not np.NaN]
        _plot_dist(pd.Series(AVG_LOS), None, label='亏损交易平均亏损额均值:{:.2f}'.format(
            np.mean(AVG_LOS)), ax=axes[0, 2])
        axes[0, 2].axvline(x=np.mean(AVG_LOS), color='#2ca02c')
        axes[0, 2].set_title('平均盈亏')
        axes[0, 2].legend()

        axes[1, 0].pie(x=[np.mean([len(m.ret) for m in metrics if hasattr(m, 'ret')]),
                          np.mean([len(m.los) for m in metrics if hasattr(m, 'los')])],
                       labels=['盈利', '亏损'],
                       colors=['#d62728', '#2ca02c'], autopct='%1.2f%%')
        axes[1, 0].set_title('盈亏次数比')

        close_benchmark = metrics[0].benchmark.kl_pd['close'] / \
            metrics[0].benchmark.kl_pd.iloc[0]['close']  # 基准收盘价
        cap = pd.concat([c.capital.capital_pd['capital_blance']
                         for c in metrics], axis=1).mean(axis=1)  # 资金变动
        cap = cap/cap.iloc[0]
        cap.plot(ax=axes[1, 1], title='资金变动', label='资金')
        close_benchmark.plot(ax=axes[1, 1], label='{}收盘价'.format(benchmark))
        axes[1, 1].legend()

        caplist = pd.Series(
            [c.capital.capital_pd['capital_blance'].iloc[-1] for c in metrics])
        _plot_dist(caplist, None, ax=axes[1, 2])
        init_cash = kwargs.pop('init_cash', Nont)
        if init_cash:
            axes[1, 2].axvline(x=init_cash, label='初始资金')
        axes[1, 2].axvline(x=caplist.mean(), color='#d68a27',
                           label='最终资金均值:{:.2f}'.format(caplist.mean()))
        axes[1, 2].set_title('最终资金')
        axes[1, 2].legend()

        money = pd.Series([c.act_sell.buy_amount.mean() for c in metrics])
        _plot_dist(money, None, label='平均买入资金:{:.2f}'.format(
            money.mean()), ax=axes[2, 0])
        money = pd.Series([c.act_sell.sell_amount.mean() for c in metrics])
        _plot_dist(money, None, label='平均卖出资金:{:.2f}'.format(
            money.mean()), ax=axes[2, 0])
        axes[2, 0].set_title('资金占用')
        axes[2, 0].legend()

        cap_mean = pd.concat([c.capital.capital_pd['cash_blance']
                              for c in metrics], axis=1).mean(axis=1)  # 账户现金均值
        cap_max = pd.concat([c.capital.capital_pd['cash_blance']
                             for c in metrics], axis=1).max(axis=1)  # 账户现金均值
        cap_mim = pd.concat([c.capital.capital_pd['cash_blance']
                             for c in metrics], axis=1).min(axis=1)  # 账户现金均值
        cap_mean.plot(
            ax=axes[2, 1], label='现金均值:{:.2f}'.format(cap_mean.iloc[-1]))
        cap_max.plot(
            ax=axes[2, 1], label='现金最大值:{:.2f}'.format(cap_max.iloc[-1]))
        cap_mim.plot(
            ax=axes[2, 1], label='现金最小值:{:.2f}'.format(cap_mim.iloc[-1]))
        axes[2, 1].set_title('账户现金')
        axes[2, 1].legend()

        # 成交次数
        times = [len(m.act_sell) for m in metrics]
        _plot_dist(pd.Series(times), None, label='成交次数均值:{:.2f}'.format(
            np.mean(times)), ax=axes[2, 2])
        axes[2, 2].set_title('成交次数')
        axes[2, 2].legend()

        if start and end:
            fig.suptitle('测试时段:{}~{}'.format(start, end))


class Indicator():
    def calc_sar(DataFrame, acceleration=0, maximum=0):
        """使用talib计算sar，即透传talib.SAR计算结果"""
        import talib
        res = talib.SAR(DataFrame.high.values,
                        DataFrame.low.values, acceleration, maximum)
        return pd.DataFrame({'SAR': res}, index=DataFrame.index)
