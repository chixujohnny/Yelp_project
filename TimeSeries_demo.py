# coding: utf-8

#
#  先选用 http://blog.csdn.net/u010414589/article/details/49622625 教程中的时间序列数据,用机器学习模型做个baseline
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import copy

dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]


# 先观察一下数据
'''
dta = pd.Series(dta) # 做成时间序列模式
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2090'))
dta.plot(figsize=(12,8)) # 最佳尺寸(20,11.5)
plt.show()'''


# 使用 RF/GBRT 预测一下最后5个年份的数据
train = [10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085]
test = [14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]

windows = [45,47,49,50,52,55] # 设置一个滑窗
plt.figure(figsize=(15, 11.5))

for w,window in enumerate(windows):

    train_new = copy.deepcopy(train)
    output = []

    for round in range(len(test)): # 预测多少年

        train_X = []
        train_y = []

        for i in range(len(train_new)-window-1):

            train_X.append(train_new[i:i+window])
            train_y.append([train_new[i+window+1]])

        train_X = np.array(train_X)
        train_y = np.array(train_y)
        model = GradientBoostingRegressor()  # GBRT模型
        # model = RandomForestRegressor()  # RF模型
        model.fit(train_X, train_y)
        test_X = train_new[-window:]
        predicted = model.predict(test_X)
        print predicted[0]
        output.append(predicted)
        train_new.append(predicted[0])

    if w <= 1:
        plt.subplot2grid((3,2), (0,0+w))
    elif w<= 3:
        plt.subplot2grid((3,2), (1,w-2))
    else:
        plt.subplot2grid((3,2), (2,w-4))

    plt.title('Time series forecasting.  Step=1, Window=' + str(window))
    # plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.plot(train + output, color='r', label='forcast') #  预测值
    plt.plot(train + test, label='original') # 原始值
    plt.legend()
    plt.grid(True)

plt.show()




