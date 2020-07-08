def init():
    import abupy
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import matplotlib.pyplot as plt
    abupy.env.init_plot_set()

    abupy.env.g_market_target = abupy.env.EMarketTargetType.E_MARKET_TARGET_CN
    # 所有任务数据强制网络更新
    abupy.env.g_data_fetch_mode = abupy.env.EMarketDataFetchMode.E_DATA_FETCH_NORMAL
    # 使用QUATAXIS本地数据作为数据源
    abupy.env.g_private_data_source = abupy.MarketBu.ABuDataFeed.QAAdvAPI

    abupy.env.g_project_rom_data_dir = r'C:\Users\GuQiang\Documents\GitHub\abu\abupy\RomDataBu'
    """忽略所有警告，默认关闭"""
    abupy.env.g_ignore_all_warnings = True
    """忽略库警告，默认打开"""
    abupy.env.g_ignore_lib_warnings = True
    """不使用自然周，自然月择时任务。参考abupy\AlphaBu\ABuPickTimeWorker.py"""
    abupy.alpha.pick_time_worker.g_natural_long_task = False

    # colors=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
