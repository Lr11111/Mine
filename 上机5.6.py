import pandas_datareader as pdr
import datetime
import numpy as np
import pandas as pd

aapl = pdr.get_data_yahoo('AAPL', start=datetime.datetime(2006, 10, 1),end=datetime.datetime(2012, 1, 1))
print(aapl.head())
# 将aapl数据框中`Adj Close`列数据赋值给变量`daily_close`
daily_close = aapl[['Adj Close']]

# 计算每日收益率
daily_pct_change = daily_close.pct_change()

# 用0填补缺失值NA
daily_pct_change.fillna(0, inplace=True)

# 查看每日收益率的前几行
print(daily_pct_change.head())

# 计算每日对数收益率
daily_log_returns = np.log(daily_close.pct_change()+1)

# 查看每日对数收益率的前几行
print(daily_log_returns.head())

# 按营业月对 `aapl` 数据进行重采样，取每月最后一项
monthly = aapl.resample('BM').apply(lambda x: x[-1])

# 计算每月的百分比变化，并输出前几行
print(monthly.pct_change().head())

# 按季度对`aapl`数据进行重采样，将均值最为每季度的数值
quarter = aapl.resample("3M").mean()

# 计算每季度的百分比变化，并输出前几行
print(quarter.pct_change().head())

# 每日收益率
daily_pct_change = daily_close / daily_close.shift(1) - 1
# 输出 `daily_pct_change`的前几行
print(daily_pct_change.head())

# 导入matplotlib
import matplotlib.pyplot as plt

# 绘制直方图
daily_pct_change.hist(bins=50)

# 显示图
plt.show()

# 输出daily_pct_change的统计摘要
print(daily_pct_change.describe())


# 计算累积日收益率
cum_daily_return = (1 + daily_pct_change).cumprod()

# 输出 `cum_daily_return` 的前几行
print(cum_daily_return.head())

# 绘制累积日收益率曲线
cum_daily_return.plot(figsize=(12,8))

# 显示绘图
plt.show()

# 将累积日回报率转换成累积月回报率
cum_monthly_return = cum_daily_return.resample("M").mean()
# 输出 `cum_monthly_return` 的前几行
print(cum_monthly_return.head())

def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
        datas = map (data, tickers)
        return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))
#现在，查看以上获取的数据：'''
print(all_data.head())

# 选取 `Adj Close` 这一列并变换数据框
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')
# 对`daily_close_px` 计算每日百分比变化
daily_pct_change = daily_close_px.pct_change()
# 绘制分布直方图
daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))
# 显示绘图结果
plt.show()

# 对 `daily_pct_change` 数据绘制散点矩阵图
pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,figsize=(12,12))
# 显示绘图结果
plt.show()

# 选取调整的收盘价
adj_close_px = aapl['Adj Close']
# 计算移动均值
moving_avg = adj_close_px.rolling(window=40).mean()
# 查看后十项结果
print(moving_avg[-10:])

# 短期的移动窗口
aapl['42'] = adj_close_px.rolling(window=40).mean()
# 长期的移动窗口
aapl['252'] = adj_close_px.rolling(window=252).mean()
# 绘制调整的收盘价，同时包含短期和长期的移动窗口均值
aapl[['Adj Close', '42', '252']].plot()
# 显示绘图结果
plt.show()

# 定义最小周期
min_periods = 75
# 计算波动率
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
# 绘制波动率曲线
vol.plot(figsize=(10, 8))
# 显示绘图结果
plt.show()

# 导入`statsmodels` 包的 `api` 模块，设置别名 `sm`
import statsmodels.api as sm
# 获取调整的收盘价数据
all_adj_close = all_data[['Adj Close']]
# 计算对数收益率
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
# 提取苹果公司数据
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')
# 提取微软公司数据
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')
# 使用 aapl_returns 和 msft_returns 创建新的数据框
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']
# 增加常数项
X = sm.add_constant(return_data['AAPL'])
# 创建模型
model = sm.OLS(return_data['MSFT'],X).fit()
# 输出模型的摘要信息
print(model.summary())

# 绘制 AAPL 和 MSFT 收益率的散点图
plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')
# 增加坐标轴
ax = plt.axis()
# 初始化 `x`
x = np.linspace(ax[0], ax[1] + 0.01)
# 绘制回归线
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)
# 定制此图
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft returns')
# 输出此图
plt.show()
'''也可以使用收益率的移动相关性对结果进行核查。只需对滚动相关性的结果调用 plot() 函 数：'''
# 绘制滚动相关性
return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()
# 显示该图
plt.show()