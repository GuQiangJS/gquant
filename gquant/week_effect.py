"""周内效应相关方法"""
from QUANTAXIS.QAIndicator.base import MAX, ABS, REF
import pandas as pd
from gquant.backtest import backtest
from faker import Faker
import datetime
from tqdm.auto import tqdm
import numpy as np


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


def up_percent(c):
    """计算按日分组后的上涨概率。"""
    return len(c[c > 0]) / len(c)


def calc_full_market(data):
    """对完整市场数据按照日期进行分组后统计"""
    return data.groupby(['weekday']).agg({
        '收盘价变化率': ['mean', 'median', up_percent],
        '日价格变化幅度': ['mean', 'median', up_percent],
    })


def calc_split_market(data):
    """对完整市场数据按照**昨天**是上涨市/下跌市+日期进行分组后统计"""
    return data.groupby(['prev_market', 'weekday']).agg({
        '收盘价变化率': ['mean', 'median', up_percent],
        '日价格变化幅度': ['mean', 'median', up_percent],
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
    low_buy_dates = low_market[(low_market['up_percent'] >= 0.5)
                               & (low_market['median'] >= 0)].index.values
    up_buy_dates = up_market[(up_market['up_percent'] >= 0.5)
                             & (up_market['median'] >= 0)].index.values
    return low_buy_dates, up_buy_dates


def calc_full_buy_opens(market):
    """
    计算完整市场下的买入方式。日价格变化幅度中 up_percent>=0.5&median>=0 的作为开盘买入

    Returns:
        开盘买入集合
    """
    market_d = market['日价格变化幅度']
    #
    return market_d[(market_d['up_percent'] >= 0.5)
                    & (market_d['median'] >= 0)].index.values


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
    low_buy_open = low_market[(low_market['up_percent'] >= 0.5)
                              & (low_market['median'] >= 0)].index.values
    up_buy_open = up_market[(up_market['up_percent'] >= 0.5)
                            & (up_market['median'] >= 0)].index.values
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
        '基准浮动盈亏(基准最后收盘/基准最先开盘)', '浮动盈亏(结算价值/初始资金)', '盈利次数', '亏损次数',
        '未结束交易购买金额', '未结束交易当前价值'
    ]].append(m.stats()[['基准最大回撤', '策略最大回撤']]).to_frame().T
    if name:
        m_report['name'] = name
        m_report.set_index('name', inplace=True)

    return m_report, m


def split_test(x, y, low_buy_dates, up_buy_dates, low_buy_opens, up_buy_opens,
               name):
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
    """
    假设现在是1.2日收盘，需要预测1.3日如何操作：
    1. 需要取1.3日数据行上的`prev_market`来拿到1.2日的market数据。也只有那1.2的market数据才是合理的。
    2. 判断1.4日是否在可买区间内，如果在可买区间内，再将1.3日标记可买。
    """
    x.loc[(x['prev_market'] == -1) & (x['nextday'].isin(low_buy_dates)),
          'buy'] = 1
    x.loc[(x['prev_market'] == 1) & (x['nextday'].isin(up_buy_dates)),
          'buy'] = 1
    x.loc[(x['prev_market'] == -1) & (~x['nextday'].isin(low_buy_dates)),
          'sell'] = 1
    x.loc[(x['prev_market'] == 1) & (~x['nextday'].isin(up_buy_dates)),
          'sell'] = 1

    def buy_func(index, row, data):
        if row['market'] == -1:
            # 下跌市
            return row['open'] if row['weekday'] in low_buy_opens else row[
                'close']
        if row['market'] == 1:
            # 上涨市
            return row['open'] if row['weekday'] in up_buy_opens else row[
                'close']

    def sell_func(index, row, data):
        if row['market'] == -1:
            # 下跌市
            return row['close'] if row['weekday'] in low_buy_opens else row[
                'open']
        if row['market'] == 1:
            # 上涨市
            return row['close'] if row['weekday'] in up_buy_opens else row[
                'open']

    split = backtest(x,
                     benchmark_pd=y,
                     buy_comm=0,
                     sell_comm=0,
                     buy_limit_comm=0,
                     sell_limit_comm=0,
                     buy_price_func=buy_func,
                     sell_price_func=sell_func)

    split_report = split.report()[[
        '基准浮动盈亏(基准最后收盘/基准最先开盘)', '浮动盈亏(结算价值/初始资金)', '盈利次数', '亏损次数',
        '未结束交易购买金额', '未结束交易当前价值'
    ]].append(split.stats()[['基准最大回撤', '策略最大回撤']]).to_frame().T

    split_report['name'] = name
    split_report.set_index('name', inplace=True)

    return split_report, split


class _A:
    def __init__(self, date, ps, fs):
        self.date = date
        self.ps = ps
        self.fs = fs

    def __eq__(self, other):
        return self.date == other.date and self.ps == other.ps and self.fs == other.fs

    def __hash__(self):
        return hash(self.date) ^ hash(self.ps) ^ hash(self.fs)

    def __str__(self):
        return 'Date:{}-PS:{}-FS:{}'.format(str(self.date), str(self.ps),
                                            str(self.fs))




def MonteCarloTest(full_data,
                   full_benchmark_data,
                   start_date=datetime.date(year=2015, month=1, day=1),
                   end_date=datetime.date(year=2019, month=12, day=31),
                   ps_min=1,
                   ps_max=3,
                   fs_min=93,
                   fs_max=366,
                   times=100000,
                   multiprocessing=False,
                   multiprocessing_kws={}):
    """
    蒙特卡洛模拟测试。
    从`start_date`~`end_date`之间随机选择一个日期，向前推`ps_min`~`ps_max`年（随机选择）作为测算数据，
    向后推`fs_min`~`fs_max`天(随机选择）作为验证数据。
    根据测算数据计算买入/卖出标准(测算方式参见calc_full_market,calc_full_buy_dates,calc_full_buy_opens及
    calc_split_market,calc_split_buy_dates,calc_split_buy_opens)，
    对验证数据进行蒙特卡洛模拟测算。*用来模拟测算随机日期是否能够跑赢基准*。
    Args:
        full_data (DataFrame): 测试用的完整数据，测试时会根据随机选择出的时间段再进行筛选。
        full_benchmark_data (DataFrame): 基准的完整数据，测试时会根据随机选择出的时间段再进行筛选。
        start_date (datetime.date): 随机选择测试时间的开始时间。默认为2015-01-01。
        end_date (datetime.date): 随机选择测试时间的截止时间。默认为2019-12-31。
        ps_min (int): 随机选择过去几年的数据作为测试数据的随机选择开始值。默认为1。
        ps_min (int): 随机选择过去几年的数据作为测试数据的随机选择截止值。默认为3。
        fs_min (int): 随机选择以后几年的数据作为验证数据的随机选择开始值。默认为93。
        fs_max (int): 随机选择以后几年的数据作为验证数据的随机选择截止值。默认为366。
        times (int): 测试次数。默认为100000。
        multiprocessing (boolean): 是否采用多进程方式处理。默认为False。
        processPoolExecutor_kws (dict): 多线程时的参数字典。
    """
    fake = Faker()

    ds = []  # 开始时间，回测几年，验证几天的集合

    pbar = tqdm(total=times, desc='准备数据')
    while len(ds) < times:
        date = fake.date_between(start_date=start_date, end_date=end_date)
        ps = fake.pyint(min_value=ps_min, max_value=ps_max)  # 过去几年的数据作为测算数据
        fs = fake.pyint(min_value=fs_min, max_value=fs_max)
        d=_A(date, ps, fs)
        if d in ds:
            continue
        ds.append(d)
        pbar.update(1)
    pbar.close()

    def _process(a):
        x_start = a.date + datetime.timedelta(days=-365 * a.ps)
        x_end = a.date + datetime.timedelta(days=-1)
        y_start = a.date
        y_end = a.date + datetime.timedelta(days=a.fs)
        data = full_data[x_start:x_end]  # 测试集
        x = full_data[y_start:y_end]  # 验证集
        y = full_benchmark_data[y_start:y_end]  # 验证基准集
        market = calc_full_market(data)
        buy_dates = calc_full_buy_dates(market)
        buy_opens = calc_full_buy_opens(market)
        r, m = full_test(x, y, buy_dates, buy_opens, name='完整')
        r['x_start'] = x_start
        r['x_end'] = x_end
        r['y_start'] = y_start
        r['y_end'] = y_end
        r['passyears'] = a.ps
        r['test_days'] = a.fs

        market_split = calc_split_market(data)
        low_buy_dates, up_buy_dates = calc_split_buy_dates(market_split)
        low_buy_open, up_buy_open = calc_split_buy_opens(market_split)
        x = full_data[y_start:y_end]
        y = full_benchmark_data[y_start:y_end]
        rs, m = split_test(x, y, low_buy_dates, up_buy_dates, low_buy_open,
                        up_buy_open, '拆分')
        rs['x_start'] = x_start
        rs['x_end'] = x_end
        rs['y_start'] = y_start
        rs['y_end'] = y_end
        rs['passyears'] = a.ps
        rs['test_days'] = a.fs
        return [r,rs]
      
    import queue
    reports = queue.Queue()  
    if multiprocessing:
        import concurrent.futures
        pbar = tqdm(total=len(ds), desc='处理中')
        with concurrent.futures.ProcessPoolExecutor(**processPoolExecutor_kws) as executor:
            future_to_url = [executor.submit(_process, d) for d in ds]
            del ds
            for future in concurrent.futures.as_completed(future_to_url):
                prime=future.result()
                reports.put(prime[0])
                reports.put(prime[1])
                future_to_url.remove(future)
                del future
                del prime
                pbar.update()
        pbar.refresh()
        pbar.close()
    else:
        pbar = tqdm(total=len(ds), desc='处理中')
        while len(ds) > 0:
            prime=_process(ds.pop())
            reports.put(prime[0])
            reports.put(prime[1])
            pbar.update(1)
        pbar.close()

    report_arr = []
    pbar = tqdm(total=reports.qsize(), desc='合并报表')
    while reports.qsize() > 0:
        report_arr.append(reports.get())
        pbar.update(1)
    pbar.close()

    report = pd.concat(report_arr).rename(columns={
        '基准浮动盈亏(基准最后收盘/基准最先开盘)': '基准浮动盈亏',
        '浮动盈亏(结算价值/初始资金)': '策略浮动盈亏'
    })
    report['跑赢基准'] = report['策略浮动盈亏'] / report['基准浮动盈亏'] - 1
    # report.style.bar(subset=['跑赢基准'], align='mid', color=['#5fba7d','#d65f5f'])
    report['跑赢基准'] = np.sign(report['跑赢基准'])
    report['是否盈利'] = np.sign(report['策略浮动盈亏'] - 1)  # 是否盈利
    return report
