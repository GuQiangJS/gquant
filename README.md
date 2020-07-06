# GQuant

主要实现对于 `abupy` 的扩展。

# 安装

`$ pip install --upgrade git+https://github.com/GuQiangJS/gquant.git`

# 主要方法

## 度量相关

- `Metrics`: `abupy.AbuMetricsBase`的扩展。
- `MetricsUtils.plot_all()`: 绘图。

## 仓位控制相关

|名称|说明|
|---|---|
|`Position_AllIn`|满仓操作。|

## 买入/卖出相关

### 卖出

|名称|说明|
|---|---|
|`SellStrategy_SAR`|SAR抛物线止损方案。当sar值大于或等于指定的对比值时卖出。|
|`SellStrategy_TDTP`|通道突破卖出方案。|
|`SellStrategy_NDay`|N日卖出方案。持有N日后，如果没有盈利，则卖出。|
|`SellStrategy_ATR`|N倍atr(止盈止损)方案。|

### 买入

|名称|说明|
|---|---|
|`BuyStrategy_TDTP`|通道突破买入方案。|