import abupy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class Metrics(abupy.AbuMetricsBase):
    """对:py:class:`abupy.AbuMetricsBase`的扩展。"""

    def _metrics_extend_stats(self):
        self.act_sell = self.action_pd[self.action_pd.action.isin(['sell'])
                                       & self.action_pd.deal.isin([True])]
        if self.act_sell.empty:
            return
        self.act_sell['sell_cost'] = self.act_sell.apply(
            lambda order: order.Price * order.Cnt, axis=1)
        self.act_sell['buy_cost'] = self.act_sell.apply(
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
            'profit'] = self.act_sell.sell_cost - self.act_sell.buy_cost - self.act_sell.sell_commission - self.act_sell.buy_commission
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
        # 每次交易相对于初始资金的收益
        self.act_sell[
            'profit_init'] = self.act_sell['profit'] / self.capital.read_cash
        # 每次交易相对于初始资金的收益百分比
        self.act_sell[
            'profit_init_hunder'] = self.act_sell['profit_init'] * 100

    def profit_series(self, **kwargs):
        return pd.Series(
            data=[
                self.sum_profit,  # 盈亏总额
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
                self.max_drawdown,
                self.act_sell['buy_cost'].mean()
            ],  # 最大回撤
            index=[
                '盈亏总额', '最终价值', '交易次数', '盈利次数', '亏损次数', '盈利比率', '每笔交易平均盈亏额',
                '盈利交易平均盈利额', '亏损交易平均亏损额', 'R', '最大回撤', '买入平均花费'
            ],
            **kwargs)


class MetricsUtils():
    def _plot_dist(series, benchmark, **kwargs):
        ax = sns.distplot(series, **kwargs)
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
        figsize = kwargs.pop('figsize', (15, 20))
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        fig, axes = plt.subplots(4, 3, figsize=figsize)

        R = pd.Series([m.R for m in metrics if hasattr(m, 'R')]).dropna()
        MetricsUtils._plot_dist(R, None, ax=axes[0, 0])
        axes[0, 0].axvline(x=R.mean(),
                           color='#d62728',
                           label='R 均值:{:.2f}'.format(R.mean()))
        axes[0, 0].set_title('R值')
        axes[0, 0].legend()

        MD = pd.Series(
            [m.max_drawdown for m in metrics if hasattr(m, 'max_drawdown')])
        MetricsUtils._plot_dist(MD, None, ax=axes[0, 1])
        axes[0, 1].axvline(x=np.mean(MD),
                           color='#d62728',
                           label='最大回撤均值:{:.2%}'.format(np.mean(MD)))
        axes[0, 1].set_title('最大回撤')
        axes[0, 1].legend()

        AVG_WIN = [
            m.avg_ret for m in metrics
            if hasattr(m, 'avg_ret') and m.avg_ret is not np.NaN
        ]
        MetricsUtils._plot_dist(pd.Series(AVG_WIN),
                                None,
                                label='盈利交易平均盈利额均值:{:.2f}'.format(
                                    np.mean(AVG_WIN)),
                                ax=axes[0, 2])
        axes[0, 2].axvline(x=np.mean(AVG_WIN), color='#d62728')
        AVG_LOS = [
            m.avg_los for m in metrics
            if hasattr(m, 'avg_los') and m.avg_los is not np.NaN
        ]
        MetricsUtils._plot_dist(pd.Series(AVG_LOS),
                                None,
                                label='亏损交易平均亏损额均值:{:.2f}'.format(
                                    np.mean(AVG_LOS)),
                                ax=axes[0, 2])
        axes[0, 2].axvline(x=np.mean(AVG_LOS), color='#2ca02c')
        axes[0, 2].set_title('平均盈亏')
        axes[0, 2].legend()

        axes[1, 0].pie(x=[
            np.mean([len(m.ret) for m in metrics if hasattr(m, 'ret')]),
            np.mean([len(m.los) for m in metrics if hasattr(m, 'los')])
        ],
            labels=['盈利', '亏损'],
            colors=['#d62728', '#2ca02c'],
            autopct='%1.2f%%')
        axes[1, 0].set_title('盈亏次数比')

        close_benchmark = metrics[0].benchmark.kl_pd['close'] / \
            metrics[0].benchmark.kl_pd.iloc[0]['close']  # 基准收盘价
        cap = pd.concat(
            [c.capital.capital_pd['capital_blance'] for c in metrics],
            axis=1).mean(axis=1)  # 资金变动
        cap = cap / cap.iloc[0]
        cap.plot(ax=axes[1, 1], title='资金变动', label='资金')
        close_benchmark.plot(ax=axes[1, 1],
                             label='{}收盘价'.format(
                                 metrics[0].benchmark.benchmark))
        axes[1, 1].legend()

        caplist = pd.Series(
            [c.capital.capital_pd['capital_blance'].iloc[-1] for c in metrics])
        MetricsUtils._plot_dist(caplist, None, ax=axes[1, 2])
        init_cash = kwargs.pop('init_cash', None)
        if init_cash:
            axes[1, 2].axvline(x=init_cash, label='初始资金')
        axes[1, 2].axvline(x=caplist.mean(),
                           color='#d68a27',
                           label='最终资金均值:{:.2f}'.format(caplist.mean()))
        axes[1, 2].set_title('最终资金')
        axes[1, 2].legend()

        money = pd.Series([c.act_sell.buy_cost.mean() for c in metrics])
        MetricsUtils._plot_dist(money,
                                None,
                                label='平均买入资金:{:.2f}'.format(money.mean()),
                                ax=axes[2, 0])
        money = pd.Series([c.act_sell.sell_cost.mean() for c in metrics])
        MetricsUtils._plot_dist(money,
                                None,
                                label='平均卖出资金:{:.2f}'.format(money.mean()),
                                ax=axes[2, 0])
        axes[2, 0].set_title('资金占用')
        axes[2, 0].legend()

        cap_mean = pd.concat(
            [c.capital.capital_pd['cash_blance'] for c in metrics],
            axis=1).mean(axis=1)  # 账户现金均值
        cap_max = pd.concat(
            [c.capital.capital_pd['cash_blance'] for c in metrics],
            axis=1).max(axis=1)  # 账户现金均值
        cap_mim = pd.concat(
            [c.capital.capital_pd['cash_blance'] for c in metrics],
            axis=1).min(axis=1)  # 账户现金均值
        cap_mean.plot(ax=axes[2, 1],
                      label='现金均值:{:.2f}'.format(cap_mean.iloc[-1]))
        cap_max.plot(ax=axes[2, 1],
                     label='现金最大值:{:.2f}'.format(cap_max.iloc[-1]))
        cap_mim.plot(ax=axes[2, 1],
                     label='现金最小值:{:.2f}'.format(cap_mim.iloc[-1]))
        axes[2, 1].set_title('账户现金')
        axes[2, 1].legend()

        # 成交次数
        times = [len(m.act_sell) for m in metrics]
        MetricsUtils._plot_dist(pd.Series(times),
                                None,
                                label='成交次数均值:{:.2f}'.format(np.mean(times)),
                                ax=axes[2, 2])
        axes[2, 2].set_title('成交次数')
        axes[2, 2].legend()

        # 买入策略占比
        d = pd.Series()
        for v in metrics:
            if d.empty:
                d = v.orders_pd['buy_factor'].value_counts()
            else:
                d = d.add(v.orders_pd['buy_factor'].value_counts(),
                          fill_value=0)
        axes[3, 0].pie(x=d,
                       labels=d.index,
                       colors=sns.color_palette("muted"),
                       autopct='%1.2f%%')
        axes[3, 0].set_title('买入策略占比')

        # 卖出入策略占比
        d = pd.Series()
        for v in metrics:
            if d.empty:
                d = v.orders_pd['sell_type_extra'].value_counts()
            else:
                d = d.add(v.orders_pd['sell_type_extra'].value_counts(),
                          fill_value=0)
        axes[3, 1].pie(x=d,
                       labels=d.index,
                       colors=sns.color_palette("muted"),
                       autopct='%1.2f%%')
        axes[3, 1].set_title('卖出入策略占比')

        if start and end:
            fig.suptitle('测试时段:{}~{}'.format(start, end))


class Indicator():
    def calc_sar(DataFrame, acceleration=0, maximum=0):
        """使用talib计算sar，即透传talib.SAR计算结果"""
        import talib
        res = talib.SAR(DataFrame.high.values, DataFrame.low.values,
                        acceleration, maximum)
        return pd.DataFrame({'SAR': res}, index=DataFrame.index)
