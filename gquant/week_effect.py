"""周内效应相关方法"""
from QUANTAXIS.QAIndicator.base import MAX, ABS, REF
import pandas as pd
from backtest import backtest


def TR(DataFrame):
    """
    真实波幅:(最高价-最低价)和昨收-最高价的绝对值的较大值和昨收-最低价的绝对值的较大值
    代码复制自https://github.com/QUANTAXIS/QUANTAXIS/blob/master/QUANTAXIS/QAIndicator/indicators.py
    """
    C = DataFrame['close']
    H = DataFrame['high']
    L = DataFrame['low']
    TR = MAX(MAX((H - L), ABS(REF(C, 1) - H)), ABS(REF(C, 1) - L))
    return pd.DataFrame({'TR': TR})


def _up_percent(c):
    """计算按日分组后的上涨概率。"""
    return len(c[c > 0])/len(c)


def calc_full_market(data):
    """对完整市场数据按照日期进行分组后统计"""
    return data.groupby(['weekday']).agg({
        '收盘价变化率': ['mean', 'median', _up_percent],
        '日价格变化幅度': ['mean', 'median', _up_percent],
    })


def calc_split_market(data):
    """对完整市场数据按照上涨市/下跌市+日期进行分组后统计"""
    return data.groupby(['market', 'weekday']).agg({
        '收盘价变化率': ['mean', 'median', _up_percent],
        '日价格变化幅度': ['mean', 'median', _up_percent],
    })


def calc_full_buy_dates(market):
    """
    计算完整市场下的买卖日期。收盘价变化率中 up_percent>=0.5&median>=0 的作为买入日期

    Returns:
        买入日集合
    """
    market_d = market['收盘价变化率']
    buy_dates = market_d[(market_d['up_percent'] >= 0.5)
                         & (market_d['median'] >= 0)].index.values
    return buy_dates


def calc_split_buy_dates(market_split):
    """
    计算不同市场下的买卖日期。

    * 下跌市收盘价变化率中 up_percent>=0.5&median>=0 的作为买入日期
    * 上涨市收盘价变化率中 up_percent>=0.5&median>=0 的作为买入日期

    Returns:
        下跌市买入日集合，上涨市买入日集合
    """
    low_market = market_split.loc[-1, '收盘价变化率']
    up_market = market_split.loc[1, '收盘价变化率']
    low_buy_dates = low_market[(low_market['up_percent'] >= 0.5) & (
        low_market['median'] >= 0)].index.values
    up_buy_dates = up_market[(up_market['up_percent'] >= 0.5) & (
        up_market['median'] >= 0)].index.values
    return low_buy_dates, up_buy_dates


def calc_full_buy_opens(market):
    """
    计算完整市场下的买入方式。日价格变化幅度中 up_percent>=0.5&median>=0 的作为开盘买入

    Returns:
        开盘买入集合
    """
    market_d = market['日价格变化幅度']
    #
    return market_d[(market_d['up_percent'] >= 0.5) & (market_d['median'] >= 0)].index.values


def calc_split_buy_opens(market_split):
    """
    计算不同市场下的买入方式。

    * 下跌市日价格变化幅度中 up_percent>=0.5&median>=0 的作为开盘买入
    * 上涨市日价格变化幅度中 up_percent>=0.5&median>=0 的作为开盘买入

    Returns:
        下跌市开盘买入集合，上涨市开盘买入集合
    """
    low_market = market_split.loc[-1, '日价格变化幅度']
    up_market = market_split.loc[1, '日价格变化幅度']
    low_buy_open = low_market[(low_market['up_percent'] >= 0.5) & (
        low_market['median'] >= 0)].index.values
    up_buy_open = up_market[(up_market['up_percent'] >= 0.5) & (
        up_market['median'] >= 0)].index.values
    return low_buy_open, up_buy_open


def full_test(x, y, buy_date, buy_open, name):
    """完整市场测试

    Args:
        x: 测试数据集
        y: 基准数据集
        buy_date: 可购买/持仓日期集合
        buy_open: 以开盘价购买的日期集合
        name: 返回报告数据表的名称

    Return:
        m_report: 报告数据表
        m (Metrics): 回测结果对象
    """

    def buy_func(index, row, data):
        if row['weekday'] in buy_open:
            return row['open']
        else:
            return row['close']

    def sell_func(index, row, data):
        if row['weekday'] in buy_open:
            return row['close']
        else:
            return row['open']

    # 下一个交易日是否符合持仓/买入标准
    x.loc[x['nextday'].isin(buy_date), 'buy'] = 1
    x.loc[~x['nextday'].isin(buy_date), 'sell'] = 1

    m = backtest(x,
                 benchmark_pd=y,
                 buy_comm=0,
                 sell_comm=0,
                 buy_limit_comm=0,
                 sell_limit_comm=0,
                 buy_price_func=buy_func,
                 sell_price_func=sell_func)

    m_report = m.report()[[
        '基准浮动盈亏(基准最后收盘/基准最先开盘)', '浮动盈亏(结算价值/初始资金)', '盈利次数', '亏损次数', '未结束交易购买金额',
        '未结束交易当前价值'
    ]].append(m.stats()[['基准最大回撤', '策略最大回撤']]).to_frame().T
    if name:
        m_report['name'] = name
        m_report.set_index('name', inplace=True)

    return m_report, m


def split_test(x, y, low_buy_dates, up_buy_dates, low_buy_opens, up_buy_opens, name):
    """
    对按照上涨市/下跌市拆分后的市场测试    

    Args:
        x: 测试数据集
        y: 基准数据集
        low_buy_dates: 下跌市买入/持有日期集合
        up_buy_dates: 上涨市买入/持有日期集合
        low_buy_opens: 下跌市以开盘价买入的日期集合
        up_buy_opens: 上涨市以开盘价买入的日期集合
        name: 返回报告数据表的名称
    """
    x.loc[(x['prev_market'] == -1) &
          (x['nextday'].isin(low_buy_dates)), 'buy'] = 1
    x.loc[(x['prev_market'] == 1) & (
        x['nextday'].isin(up_buy_dates)), 'buy'] = 1
    x.loc[(x['prev_market'] == -1) &
          (~x['nextday'].isin(low_buy_dates)), 'sell'] = 1
    x.loc[(x['prev_market'] == 1) & (
        ~x['nextday'].isin(up_buy_dates)), 'sell'] = 1

    def buy_func(index, row, data):
        if row['market'] == -1:
            # 下跌市
            return row['open'] if row['weekday'] in low_buy_opens else row['close']
        if row['market'] == 1:
            # 上涨市
            return row['open'] if row['weekday'] in up_buy_opens else row['close']

    def sell_func(index, row, data):
        if row['market'] == -1:
            # 下跌市
            return row['close'] if row['weekday'] in low_buy_opens else row['open']
        if row['market'] == 1:
            # 上涨市
            return row['close'] if row['weekday'] in up_buy_opens else row['open']

    split = backtest(x,
                     benchmark_pd=y,
                     buy_comm=0,
                     sell_comm=0,
                     buy_limit_comm=0,
                     sell_limit_comm=0,
                     buy_price_func=buy_func,
                     sell_price_func=sell_func)

    split_report = split.report()[[
        '基准浮动盈亏(基准最后收盘/基准最先开盘)', '浮动盈亏(结算价值/初始资金)', '盈利次数', '亏损次数', '未结束交易购买金额',
        '未结束交易当前价值'
    ]].append(split.stats()[['基准最大回撤', '策略最大回撤']]).to_frame().T

    split_report['name'] = name
    split_report.set_index('name', inplace=True)

    return split_report, split
