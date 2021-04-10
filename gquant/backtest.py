import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from empyrical import stats


def backtest(data, init_cash=10000, **kwargs):
    """
    模拟回测
    Args:
        data (pd.DataFrame): 数据源。以日期为index，其中包含buy和sell列。分别以1标识为买入和卖出。
        init_cash: 初始资金。默认10000。
        buy_price_func: 就算购买时使用的数据的方法。默认接受`index`,`row`,`data`三个数据，`index`为data的当前位置,`row`为当天数据。当此参数有值时会忽略`buy_col`参数。
        buy_col (str): 购买时使用的列。默认为'open'。
        buy_limit (int): 最小购买数量。默认100。
        buy_tax: 购买时印花税率。默认为0.0003(万3)。
        buy_comm: 购买时佣金比率。默认为0.00025(万2.5)。
        buy_limit_comm: 购买时最低佣金金额。默认为5。
        sell_price_func: 就算卖出时使用的数据的方法。默认接受`index`,`row`,`data`三个数据，`index`为data的当前位置,`row`为当天数据。当此参数有值时会忽略`sell_col`参数。
        sell_col (str): 卖出时使用的列。默认为'close'。
        sell_tax: 卖出时印花税率。默认为0.0003(万3)。
        sell_comm: 卖出时佣金比率。默认为0.00025(万2.5)。
        sell_limit_comm: 卖出时最低佣金金额。默认为5。
        benchmark_pd: 基准数据。默认为data。
    Returns:
        Metrics对象。
    """
    buy_df = []
    sell_df = []
    cash = init_cash
    hold_amount = 0
    buy_col = kwargs.pop("buy_col", 'open')
    buy_limit = kwargs.pop('buy_limit', 100)
    buy_tax = kwargs.pop('buy_tax', 0.0003)
    buy_comm = kwargs.pop('buy_comm', 0.00025)
    buy_limit_comm = kwargs.pop('buy_limit_comm', 5)
    sell_col = kwargs.pop("sell_col", 'close')
    sell_tax = kwargs.pop('sell_tax', 0.0003)
    sell_comm = kwargs.pop('sell_comm', 0.00025)
    sell_limit_comm = kwargs.pop('sell_limit_comm', 5)
    benchmark_pd = kwargs.pop('benchmark_pd', data)
    buy_price_func = kwargs.pop('buy_price_func', None)
    sell_price_func = kwargs.pop('sell_price_func', None)
    for index, row in tqdm(data.iterrows()):
        if hold_amount == 0 and row['buy'] == 1:
            if buy_price_func != None:
                buy_price = buy_price_func(index, row, data)
            else:
                buy_price = row[buy_col]
            amount = _calc_buy_amount(cash, buy_price, limit=buy_limit)
            if amount > 0:
                hold_amount = amount
                buy_com = _calc_commission(
                    amount, buy_price, tax=buy_tax, comm=buy_comm, limit_comm=buy_limit_comm)
                funds = cash-buy_price*amount-buy_com  # 购买后剩余资金
                logging.debug('{}:{:.2f}-{:.2f}-{:.2f}={:.2f}'.format(row.name,
                                                                      cash, buy_price*amount, buy_com, funds))
                buy_df.append(pd.DataFrame({'buy_date': row.name, 'buy_price': buy_price,
                                            'buy_amount': amount, 'buy_comm': buy_com, 'buy_cash': cash, 'buy_funds': funds}, index=[0]))
                cash = cash-buy_price*amount-buy_com
                continue
        if hold_amount > 0 and row['sell'] == 1:
            if sell_price_func != None:
                sell_price = sell_price_func(index, row, data)
            else:
                sell_price = row[sell_col]
            sell_com = _calc_commission(
                hold_amount, sell_price, tax=sell_tax, comm=sell_comm, limit_comm=sell_limit_comm)
            logging.debug('{}:{:.2f}+{:.2f}-{:.2f}={:.2f}'.format(row.name, cash,
                                                                  sell_price*hold_amount, sell_com, cash+sell_price*hold_amount-sell_comm))
            fund = cash+sell_price*hold_amount-sell_com  # 卖出后剩余资金
            sell_df.append(pd.DataFrame({'sell_date': row.name, 'sell_price': sell_price,
                                         'sell_amount': hold_amount, 'sell_comm': sell_com, 'sell_cash': cash, 'sell_funds': fund}, index=[0]))
            cash = cash+sell_price*hold_amount-sell_com
            hold_amount = 0

    buy_df = pd.concat(buy_df).reset_index().drop(columns='index')
    sell_df = pd.concat(sell_df).reset_index().drop(columns='index')

    return Metrics(buy_df, sell_df, init_cash, cash, benchmark_pd)


class Metrics():
    def __init__(self, buy_pd, sell_pd, init_cash, cash, benchmark_pd, market_trade_year=250):
        """
        Args:
            buy_pd: 以数字顺序作为index的购买记录的DataFrame。其中包含'buy_price','buy_amount','buy_date'列。
            sell_pd: 以数字顺序作为index的卖出记录的DataFrame。其中包含'sell_price','sell_amount','sell_date','sell_funds'列。
            init_cash: 初始资金。
            cash: 剩余资金。
            benchmark_pd: 基准数据源。
            market_trade_year: 市场中1年交易日，默认250日。
        """
        self.cash = cash
        self.buy_pd = buy_pd
        self.sell_pd = sell_pd
        self.init_cash = init_cash
        self.benchmark_pd = benchmark_pd
        self.x_df = self.buy_pd.join(self.sell_pd)
        self.x_df['sell_cost'] = self.x_df['sell_price'] * \
            self.x_df['sell_amount']
        self.x_df['sell_cost_comm'] = self.x_df['sell_funds'] - \
            self.x_df['sell_cash']  # 含交易费的卖出实际收入
        self.x_df['buy_cost'] = self.x_df['buy_price']*self.x_df['buy_amount']
        self.x_df['buy_cost_comm'] = self.x_df['buy_cash'] - \
            self.x_df['buy_funds']  # 含交易费的买入实际花费
        self.x_df['profit'] = self.x_df['sell_cost']-self.x_df['buy_cost']
        self.x_df['profit_comm'] = self.x_df['sell_cost_comm'] - \
            self.x_df['buy_cost_comm']  # 含交易费的实际收益情况

        self.buy_pd.set_index('buy_date', inplace=True)
        self.buy_pd.index.rename('date', inplace=True)
        self.sell_pd.set_index('sell_date', inplace=True)
        self.sell_pd.index.rename('date', inplace=True)

        # 收益数据
        self.benchmark_returns = np.round(
            self.benchmark_pd.close.pct_change(), 3)
        self.algorithm_returns = np.round(
            self.sell_pd['sell_funds'].pct_change(), 3)
        # 收益cum数据
        self.algorithm_cum_returns = stats.cum_returns(self.algorithm_returns)
        self.benchmark_cum_returns = stats.cum_returns(self.benchmark_returns)

        # 最后一日的cum return
        self.benchmark_period_returns = self.benchmark_cum_returns[-1]
        self.algorithm_period_returns = self.algorithm_cum_returns[-1]

        # 交易天数
        self.num_trading_days = len(self.benchmark_returns)

        # 年化收益
        self.algorithm_annualized_returns = \
            (market_trade_year / self.num_trading_days) * \
            self.algorithm_period_returns
        self.benchmark_annualized_returns = \
            (market_trade_year / self.num_trading_days) * \
            self.benchmark_period_returns

        # 最大回撤
        self.algorithm_max_drawdown = stats.max_drawdown(
            self.algorithm_returns.values)
        self.benchmark_max_drawdown = stats.max_drawdown(
            self.benchmark_returns.values)

    def stats(self):
        d = {
            '基准收益': self.benchmark_period_returns,
            '策略收益': self.algorithm_period_returns,
            '基准年化收益': self.benchmark_annualized_returns,
            '策略年化收益': self.algorithm_annualized_returns,
            '基准最大回撤': self.benchmark_max_drawdown,
            '策略最大回撤': self.algorithm_max_drawdown
        }
        return pd.Series(d, index=d.keys())

    def report(self):

        d = {
            '剩余现金': '{:.2f}'.format(self.cash),
            '交易次数': '{:.0f}'.format(self.x_df.shape[0]),
            '未结束交易次数': '{:.0f}'.format(self.x_df[self.x_df['sell_price'].isna()].shape[0]),
            '未结束交易购买金额': '{:.2f}'.format(self.x_df[self.x_df['sell_price'].isna()].buy_cost.sum()),
            '盈利次数': '{:.0f}'.format(self.x_df[self.x_df['profit'] > 0].shape[0]),
            '亏损次数': '{:.0f}'.format(self.x_df[self.x_df['profit'] < 0].shape[0]),
            '盈利次数占比': '{:.2%}'.format(self.x_df[self.x_df['profit'] > 0].shape[0]/self.x_df.shape[0]),
            '盈利(含交易费)次数': '{:.0f}'.format(self.x_df[self.x_df['profit_comm'] > 0].shape[0]),
            '亏损(含交易费)次数': '{:.0f}'.format(self.x_df[self.x_df['profit_comm'] < 0].shape[0]),
            '盈利(含交易费)次数占比': '{:.2%}'.format(self.x_df[self.x_df['profit_comm'] > 0].shape[0]/self.x_df.shape[0]),
            '盈利交易平均获利': '{:.2f}'.format(self.x_df[self.x_df['profit'] > 0].profit.mean()),
            '亏损交易平均亏损': '{:.2f}'.format(self.x_df[self.x_df['profit'] < 0].profit.mean()),
            '盈利(含交易费)交易平均获利': '{:.2f}'.format(self.x_df[self.x_df['profit_comm'] > 0].profit.mean()),
            '亏损(含交易费)交易平均亏损': '{:.2f}'.format(self.x_df[self.x_df['profit_comm'] < 0].profit.mean()),
            '盈亏总额': '{:.2f}'.format(self.x_df['profit'].sum()),
            '手续费总额': '{:.2f}'.format(self.x_df['sell_comm'].sum()+self.x_df['buy_comm'].sum()),
            '手续费均值': '{:.2f}'.format((self.buy_pd['buy_comm'].mean()+self.sell_pd['sell_comm'].mean())/2),
            '最大盈利%': '{:.2%}'.format((self.x_df['profit']/self.x_df['buy_cost']).max()),
            '最大亏损%': '{:.2%}'.format((self.x_df['profit']/self.x_df['buy_cost']).min()),
            '最大(含交易费)盈利%': '{:.2%}'.format((self.x_df['profit_comm']/self.x_df['buy_cost']).max()),
            '最大(含交易费)亏损%': '{:.2%}'.format((self.x_df['profit_comm']/self.x_df['buy_cost']).min()),
        }
        return pd.Series(d, index=d.keys())

    def plot_cash(self):
        self.algorithm_cum_returns.plot(label='策略收益')
        self.benchmark_cum_returns.plot(label='基准收益')
        plt.legend()
        plt.show()


def _calc_buy_amount(cash, price, limit=100):
    """计算可买数量，对limit取整"""
    return int(cash/price/limit)*limit


def _calc_commission(trade_cnt, price, tax=0.0003, comm=0.00025, limit_comm=5):
    """
    a股计算交易费用：印花税＋佣金
    Args:
        trade_cnt（int）:交易的股数。
        price :每股金额。
        tax: 印花税率。默认为0.0003(万3)。
        comm: 佣金比率。默认为0.00025(万2.5)。
        limit_comm: 最低佣金金额。默认为5。
    """
    cost = trade_cnt * price
    # 印花税万3，
    tax = cost * tax
    # 佣金万2.5
    commission = cost * comm
    # 佣金最低5
    commission = commission if commission > limit_comm else limit_comm
    commission += tax
    return commission
