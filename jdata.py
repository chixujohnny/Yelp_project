# coding: utf-8


# ------------------------------------------- #
#                                             #
#                京东JData算法大赛              #
#                                             #
#                Author: John                 #
#                Date:   2017.3.23            #
#                                             #
# ------------------------------------------- #


import pprint
import pandas as pd
import numpy as np
import chardet
import sklearn.preprocessing as preprocessing


# 整合三个月所有Action数据,对时间降序
def Combin_Action_Data():

    df_02 = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_201602.csv')
    df_03_0 = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_201603.csv')
    df_03_1 = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_201603_extra.csv')
    df_04 = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_201604.csv')

    df_02 = df_02.append(df_03_0)
    df_02 = df_02.append(df_03_1)
    df_02 = df_02.append(df_04)

    df = df_02.sort(['time'])
    df.to_csv('/Users/John/Desktop/JData_dataset/JData_Action.csv', index=False)

    return 0
# Combin_Action_Data()


# 将time列扩展成month、day、hour、minute
def Split_Time():

    f = open('/Users/John/Desktop/JData_dataset/JData_Action_0.csv', 'w')
    f.write('user_id,sku_id,time,model_id,type,cate,brand,month,day,hour,minute\n')
    lines = open('/Users/John/Desktop/JData_dataset/JData_Action.csv', 'r').readlines()

    for line in lines[1:]:

        timestamp = line.split(',')[2]
        date = timestamp.split(' ')[0]
        time = timestamp.split(' ')[1]
        month = date.split('-')[1]
        day = date.split('-')[2]
        hour = time.split(':')[0]
        minute = time.split(':')[1]

        f.write(line.replace('\n','') + ',' + month + ',' + day + ',' + hour + ',' + minute + '\n')

    # 删除time列,移动month、day、hour、minute列
    df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_0.csv')
    df = df.drop(['time'], axis=1)
    month = df.pop('month')
    day = df.pop('day')
    hour = df.pop('hour')
    minute = df.pop('minute')
    df.insert(2, 'month', month)
    df.insert(3, 'day', day)
    df.insert(4, 'hour', hour)
    df.insert(5, 'minute', minute)

    df.to_csv('/Users/John/Desktop/JData_dataset/JData_Action_1.csv', index=False)
    return 0
# Split_Time()


# baseline: 筛选4.14加入购物车的user-sku
def Artificial_Rules_0():

    df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_1.csv')
    df_cart = df[df.month==4][df.day==14][df.type==2]
    df_cart_US = df_cart.loc[:, ['user_id','sku_id']]
    df_cart_US = df_cart_US.drop_duplicates(['user_id']) # 把user_id列进行去重
    df_cart_US.to_csv('/Users/John/Desktop/JData_dataset/JData_Artificial_Rules_0.csv', index=False)

    return 0
# Artificial_Rules_0()


# 数据清洗,去除噪音点
def Data_Clean():

    df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_1.csv')
    # df_user_valuecounts = df['user_id'].value_counts()
    # df_user_valuecounts.to_csv('/Users/John/Desktop/JData_dataset/User_ValueCounts.csv')
    print (df['user_id'].value_counts()<=10)==True
    print (df['user_id'].value_counts()<=10)==True.index
# Data_Clean()


# 特征工程
def Feature_Engineering(df):

    scaler = preprocessing.StandardScaler() # 将数据规则化到[-1,1]

    def Rate(buy, view): # 计算转化率,buy和view均为dict格式

        rate = {}
        for item in buy:

            if view.has_key(item) == False:
                # print item
                continue
            else:
                rate[item] = float(buy[item])/float(view[item])

        return pd.DataFrame.from_dict(rate, 'index')

    # 近7天,浏览转化率
    # df_user_view = df.user_id[ df['type']==1 ].value_counts().to_dict() # 用户历史浏览数,dict格式
    # df_user_buy = df.user_id[ df['type']==4 ].value_counts().to_dict() # 用户历史下单数,dict格式
    # rate = Rate(df_user_buy, df_user_view)
    # rate.columns = ['buyView_rate_7days']
    # # print type(rate.describe()) # 看一下是什么类型的数据
    # rate['user_id'] = rate.index # 把index作为一个column
    # print rate
    # df_f = df.iloc[:,:2].drop_duplicates() # (user_id--sku_id)对,去重
    # df_f = pd.merge(df_f.iloc[:,:2], rate, how='left', on='user_id')

    # 近3天,浏览转化率
    # df_user_view = df.user_id[ df.type==1 ].value_counts().to_dict()
    # df_user_buy = df.user_id[ df.type==4 ].value_counts().to_dict()
    # rate = Rate(df_user_buy, df_user_view)
    # rate.columns = ['buyView_rate_3days']
    # rate['user_id'] = rate.index

    # 近7天,加购转化率
    # df_user_cart = df.user_id[df['type'] == 2].value_counts().to_dict()
    # df_user_buy = df.user_id[df['type'] == 4].value_counts().to_dict()
    # rate = Rate(df_user_buy, df_user_cart)
    # rate.columns = ['buyCart_rate_7days']
    # rate['user_id'] = rate.index

    # 近3天,加购转化率
    # df_user_cart = df.user_id[ df.type==2 ].value_counts().to_dict()
    # df_user_buy = df.user_id[ df.type==4 ].value_counts().to_dict()
    # rate = Rate(df_user_buy, df_user_cart)
    # rate.columns = ['buyCart_rate_3days']
    # rate['user_id'] = rate.index

    # 商品历史购买数(需要规则化规则化,要不数太大了)
    # df_sku_viewNum = df.sku_id[ df.type==4 ].value_counts() # Seires格式
    # df_sku_viewNum = df_sku_viewNum.to_frame()
    # df_sku_viewNum_scale_param = scaler.fit(df_sku_viewNum) # model格式
    # df_sku_viewNum['sku_id'] = df_sku_viewNum.index
    # df_sku_viewNum['buyNum'] = scaler.fit_transform(df_sku_viewNum, df_sku_viewNum_scale_param)
    # feature = df_sku_viewNum
    # print df_sku_viewNum

    # 商品历史浏览数(规则化)
    # df_sku_viewNum = df.sku_id[ df.type==1 ].value_counts()
    # df_sku_viewNum = df_sku_viewNum.to_frame()
    # df_sku_viewNum_scale_param = scaler.fit(df_sku_viewNum)
    # df_sku_viewNum['sku_id'] = df_sku_viewNum.index
    # df_sku_viewNum['viewNum'] = scaler.fit_transform(df_sku_viewNum, df_sku_viewNum_scale_param)
    # feature = df_sku_viewNum

    # 商品历史加购数(规则化)
    df_sku_cartNum = df.sku_id[ df.type==2 ].value_counts()
    df_sku_cartNum = df_sku_cartNum.to_frame()
    df_sku_cartNum_scale_param = scaler.fit(df_sku_cartNum)
    df_sku_cartNum['sku_id'] = df_sku_cartNum.index
    df_sku_cartNum['cartNum'] = scaler.fit_transform(df_sku_cartNum, df_sku_cartNum_scale_param)
    feature = df_sku_cartNum

    # 商品从购物车删除数(规则化)
    # df_sku_dropCartNum = df.sku_id[ df.type==3 ].value_counts()
    # df_sku_dropCartNum = df_sku_dropCartNum.to_frame()
    # df_sku_dropCartNum_scale_param = scaler.fit(df_sku_dropCartNum)
    # df_sku_dropCartNum['sku_id'] = df_sku_dropCartNum.index
    # df_sku_dropCartNum['dropCartNum'] = scaler.fit_transform(df_sku_dropCartNum, df_sku_dropCartNum_scale_param)
    # feature = df_sku_dropCartNum

    # 商品关注数(规则化)
    # df_sku_likeNum = df.sku_id[ df.type==5 ].value_counts()
    # df_sku_likeNum = df_sku_likeNum.to_frame()
    # df_sku_likeNum_scale_param = scaler.fit(df_sku_likeNum)
    # df_sku_likeNum['sku_id'] = df_sku_likeNum.index
    # df_sku_likeNum['likeNum'] = scaler.fit_transform(df_sku_likeNum, df_sku_likeNum_scale_param)
    # feature = df_sku_likeNum


    return feature


# df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_1.csv')
# df0 = df[ (df['month']==4) & (df['day']>=3) & (df['day']<=9) ]
# df1 = df[ (df['month']==4) & (df['day']>=10) & (df['day']<=14) ]
# df2 = df[ (df['month']==4) & (df['day']>=7) & (df['day']<=9) ]
# df0.to_csv('/Users/John/Desktop/JData_dataset/JData_Action_403_409.csv', index=False)
# df1.to_csv('/Users/John/Desktop/JData_dataset/JData_Action_410_414.csv', index=False)
# df2.to_csv('/Users/John/Desktop/JData_dataset/JData_Action_407_409.csv', index=False)
# df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_403_409.csv')
# df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_407_409.csv')
# df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_410_414.csv')

# df_f = Feature_Engineering(df)
# df_f.to_csv('/Users/John/Desktop/JData_dataset/JData_FE.csv', index=False)

# df_f = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_FE.csv')
# rate = Feature_Engineering(df)
# df_f = pd.merge(df_f, rate, how='left', on='user_id')
# df_f = df_f.fillna(0)
# print df_f
# df_f.to_csv('/Users/John/Desktop/JData_dataset/JData_FE.csv', index=False)

# 放置目标值
# df_buy = df[ df.type==4 ]
# df_buy = df_buy.iloc[:, :2].drop_duplicates()
# df_buy['y'] = pd.Series() # 增加一列
# df_buy = df_buy.fillna(1) # 将目标值y置1
#
# df_modelData = pd.merge(df_f, df_buy, how='left', on=['user_id','sku_id'])
# df_modelData = df_modelData.fillna(0)
# df_modelData.to_csv('/Users/John/Desktop/JData_dataset/JData_modelData.csv', index=False)

# ------------------------------------------- #
# 做到这里的时候发现一个问题
# 在7天内发生过交互行为的user-sku共有79w条记录
# 但是实际有购买行为的只有3~4k条
# 数据偏移严重!
# 下面要做的是将历史数据中从不买东西的user去掉
# ------------------------------------------- #

# print pd.DataFrame(df.user_id.drop_duplicates(), columns=['user_id']) # 看看一共有多少用户
# df_buy_user = pd.DataFrame(df.user_id[ df.type==4 ].drop_duplicates(), columns=['user_id']) # 取出发生过购物行为的user_id,并去重
# print df_buy_user

# ------------------------------------------- #
# 10w的用户总数中竟只有将近3w用户发生过购物行为!
# 将发生过购物行为的用户保存一下
# 并在训练集中剔除没发生购物行为的用户
# ------------------------------------------- #

# df_buy_user.to_csv('/Users/John/Desktop/JData_dataset/JData_发生过购物行为的用户.csv', index=False)
# df_buy_user = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_发生过购物行为的用户.csv')
# df_modelData = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_modelData.csv')
# print df_buy_user # 看一下发生过购买行为的用户数量
# print df_modelData # 先看一下建模数据的数据量
# df = pd.merge(df_buy_user, df_modelData, how='left', on='user_id')
# df.to_csv('/Users/John/Desktop/JData_dataset/JData_modelData_剔除未发生购买行为用户.csv', index=False)

# ------------------------------------------- #
# 数据量从79w缩减到29w
# 现在发现特征还是有点少,再搞点特征
# ------------------------------------------- #

df = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_Action_1.csv')
df_modelData = pd.read_csv('/Users/John/Desktop/JData_dataset/JData_modelData_剔除未发生购买行为用户.csv')
df_modelData = df_modelData.drop(['cartNum'], axis=1)
feature = Feature_Engineering(df)
df_modelData = pd.merge(df_modelData, feature, how='left', on='sku_id')
df_modelData = df_modelData.fillna(0)
print df_modelData
df_modelData.to_csv('/Users/John/Desktop/JData_dataset/JData_modelData_剔除未发生购买行为用户.csv', index=False)

