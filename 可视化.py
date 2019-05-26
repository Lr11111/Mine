import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
from pylab import mpl

mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
sh=ts.get_k_data(code='sh',ktype='D',autype='qfq',start='1990-12-20')
print(sh.head(5))
sh.index=pd.to_datetime(sh.date)
sh['close'].plot(figsize=(12,6))
plt.title('Trend chart for SH stocks from 1990 to NOW')
plt.xlabel('date')
plt.show()

print(sh.describe().round(2))
print(sh.count())
print(sh.close.mean())
print(sh.open.std())
sh.loc['2007-01-01':]['volume'].plot(figsize=(12,6))
plt.title('from 01/01/2016')
plt.show()

ma_day=[20,52,252]
for ma in ma_day:
    column_name='%sday mean'%(str(ma))
    sh[column_name]=sh['close'].rolling(ma).mean()
sh.tail(3)

sh.loc['2007-01-01':][['close','20day mean','52day mean','252day mean']].plot(figsize=(12,6))
plt.title('2007 to NOW the trend of CHN Stock Market')
plt.xlabel('date')
plt.show()

sh['daily profit']=sh['close'].pct_change()
sh['daily profit'].loc['2006-01-01':].plot(figsize=(12,6))
plt.xlabel('date')
plt.ylabel('daily profit')
plt.title('From 2006 to now daily profit')
plt.show()

sh['daily profit'].loc['2006-01-01':].plot(figsize=(12,4),linestyle='--',marker='o',color='orange')
plt.xlabel('date')
plt.show()

stocks={'上证指数':'sh','深证指数':'sz','沪深300':'hs300','上证50':'sz50','中小指数':'zxb','创业板':'cyb'}

def return_risk(stocks,startdate='2006-1-1'):#j计算风险
    close=pd.DataFrame()
    for stock in stocks.values():
        close[stock]=ts.get_k_data(stock,ktype='D',autype='qfq',start=startdate)['close']
    tech_rets=close.pct_change()[1:] #算收益
    rets=tech_rets.dropna() #去空值
    ret_mean=rets.mean()*100#平均值
    ret_std=rets.std()*100#标准差
    return  ret_mean,ret_std

def plot_return_risk():
    ret,vol=return_risk(stocks)
    color=np.array([0.18,0.96,0.75,0.3,0.9,0.5])
    plt.scatter(ret,vol,marker='o',c=color,s=500,cmap=plt.get_cmap('Spectral'))
    plt.xlabel("日收益率均值%")
    plt.ylabel("标准差%")
    for label,x,y in zip(stocks.keys(),ret,vol):
        plt.annotate(label,xy=(x,y),xytext=(20,20),textcoords="offset points",ha="right",va="bottom",bbox=dict(boxstyle=
    'round,pad=0.5',fc='yellow',alpha=0.5),arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0"))

stocks={'上证指数':'sh','深证指数':'sz','沪深300':'hs300','上证50':'sz50','中小指数版':'zxb','创业指数板':'cyb'}
plot_return_risk()

stocks={'中国平安':'601318','格力电器':'000651','徐工机械':'000425','招商银行':'600036','恒生电子':'600570','贵州茅台':'600519'}
startdate='2008-1-1'
plot_return_risk()

df=ts.get_k_data('sh',ktype='D',autype='qfq',start='2006-01-01')
df.index=pd.to_datetime(df.date)
tech_rets=df.close.pct_change()[1:]
rets=tech_rets.dropna()
print(rets.head(100))
print(rets.quantile(0.05))

def monte_carlo(start_price,days,mu,sigma):
    dt=1/days
    price=np.zeros(days)
    price[0]=start_price
    shock=np.zeros(days)
    drift=np.zeros(days)

    for x in range(1,days):
        shock[x]=np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x]=mu*dt
        price[x]=price[x-1]+(price[x-1]*(drift[x]+shock[x]))
    return price

runs=10000
start_price=2641.34
days=365
mu=rets.mean()
sigma=rets.std()
simulations=np.zeros(runs)

for run in range(runs):
    simulations[run]=monte_carlo(start_price,days,mu,sigma)[days-1]
q=np.percentile(simulations,1)

plt.figure(figsize=(10,6))
plt.hist(simulations,bins=100,color='grey')
plt.figtext(0.6,0.8,s='初始价格:%.2f'%start_price)
plt.figtext(0.6,0.7,'预期价格均值:%.2f'%simulations.mean())
plt.figtext(0.15,0.6,'q(0.99:%.2f)'%q)
plt.axvline(x=q,linewidth=6,color='r')
plt.title('经过%s天上证指数的模特卡罗模拟后的价格分布图'%days,weight='bold')
plt.show()

from time import time
np.random.seed(2018)
t0=time()
s0=2641.34
T=1.0;
r=0.05;
sigma=rets.std()
M=50;
dt=T/M;
I=250000
s=np.zeros(I)
s[0]=s0
for t in range(1,M+1):
    z=np.random.standard_normal(I)
    s[t]=s[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
s_m=np.sum(s[-1])/I
tnp1=time()-t0
print('经过250000次模拟，得出1年后上证指数的预期平均收盘价为:%.2f'%s_m)