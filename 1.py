import matplotlib.pyplot as plt  # 提供类matlab里绘图框架
import numpy as np
import pandas as pd
import tushare as ts

# 获取数据
s_1 = '000157'  # 全聚德
s_2 = '600031'  # 光明乳业
sdate = '2016-01-01'  # 起止日期
edate = '2016-12-31'
df_s1 = ts.get_k_data(s_1,start=sdate,end=edate).sort_index(axis=0, ascending=True)  # 获取历史数据
df_s2 = ts.get_k_data(s_2, start=sdate, end=edate).sort_index(axis=0, ascending=True)
df = pd.concat([df_s1.open, df_s2.open],axis=1,keys=['s1_open','s2_open'])  # 合并
df.ffill(axis=0,inplace=True)  # 填充缺失数据
df.to_csv('s12.csv')

# pearson方法计算相关性
corr = df.corr(method='pearson', min_periods=1)
print(corr)

df.s1_open.plot(figsize=(16,12))
df.s2_open.plot(figsize=(16,12))
plt.show()

data=pd.read_csv('directory.csv')
print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())
print(data[data['City'].isnull()])

def fill_na(x):
    return x
data['City']=data['City'].fillna(fill_na(data['State/Province']))
data['Country'][data['Country']=='TW']='CN'
print(data['Country'][data['Country']=='TW'])