# coding: utf-8


import pandas as pd
import json
import matplotlib.pyplot as plt
import sys
import nltk
import chardet
import numpy as np
import datetime
import time
from scipy import interpolate
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import copy


##############################
#  更新数据结构(Dataframe格式)  #
##############################
from nltk.util import pr


def Review_DataStructure_Pd(yelp_review_path, yelp_business_path, review_csv):

    # 先将 business-category 存到一张哈希表中
    business_category_dict = {}
    lines = open(yelp_business_path, 'r').readlines()
    for line in lines:
        line_json = json.loads(line)
        bid = line_json['business_id']
        category = line_json['categories']
        business_category_dict[bid] = category # {'abc':['Chinese','Restaurants']}

    f = open(review_csv, 'w')
    f.write('User_id,' + 'Business_id,' + 'year,' + 'month,' + 'day,' + 'category,' + 'review' + '\n')
    lines = open(yelp_review_path, 'r').readlines()
    for line in lines:
        line_json = json.loads(line)
        uid = line_json['user_id'].encode('utf-8')
        bid = line_json['business_id'].encode('utf-8')
        year = line_json['date'].split('-')[0].encode('utf-8')
        month = line_json['date'].split('-')[1].encode('utf-8')
        day = line_json['date'].split('-')[2].encode('utf-8')
        category = '"' + (','.join(business_category_dict[bid])).encode('utf-8') + '"' # "Chinese,Restuarants"
        review = '"' + line_json['text'].encode('utf-8').replace(('\n' or '\n\n' or ','), '.').replace('"', "'") + '"'
        f.write(uid + ',' + bid + ',' + year + ',' + month + ',' + day + ',' + category + ',' + review + '\n')

    return 0

yelp_review_path = '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
yelp_business_path = '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
review_csv = '/Users/John/Desktop/Yelp_dataset/Review.csv'
# Review_DataStructure_Pd(yelp_review_path, yelp_business_path, review_csv)


# 各年月评论分布
'''
g = df.groupby(['year', 'month'])
df = pd.DataFrame(g.count()['review'])

fig = plt.figure()
fig.set(alpha=0.2) # 设定图标颜色 alpha 参数
df.review.plot(kind='bar') # 柱形图
plt.xlabel('Date')
plt.ylabel('Number of review')
plt.title('The number of comments in the distribution')
plt.show()'''


# 用户评论数量分布
'''
fig = plt.figure()
fig.set(alpha=0.2) # 设定图标颜色 alpha 参数

df = df['User_id'].value_counts().value_counts()
df.plot(kind='bar')

plt.xlabel('Number of review')
plt.ylabel('Number')
plt.title('User comment number distribution')
plt.show()'''


# 按服务类别筛选review
# df = pd.read_csv(review_csv)
# df_Restaurants = df[df['category'].str.contains('Restaurants',na=False)]
# df_Nightlife = df[df['category'].str.contains('Nightlife',na=False)]
# df_Shopping = df[df['category'].str.contains('Shopping',na=False)]
# df_Food = df[df['category'].str.contains('Food',na=False)]

# df_Restaurants.to_csv('/Users/John/Desktop/Yelp_dataset/Restaurants/df_data.csv', index=False)
# df_Nightlife.to_csv('/Users/John/Desktop/Yelp_dataset/Nightlife/df_data.csv', index=False)
# df_Shopping.to_csv('/Users/John/Desktop/Yelp_dataset/Shopping/df_data.csv', index=False)
# df_Food.to_csv('/Users/John/Desktop/Yelp_dataset/Food/df_data.csv', index=False)


# 挖掘该 category 所有的feature,取top50并排序
def Dig_Feature(df):

    review = []
    for text in df['review']:
        review.append(text)

    # 进度条
    def View_Bar(flag, sum):
        rate = float(flag) / sum
        rate_num = rate * 100
        if flag % 15.0 == 0:
            print '\r%.2f%%: ' % (rate_num),  # \r%.2f后面跟的两个百分号会输出一个'%'
            sys.stdout.flush()

    # 词性标注
    def Word_Tagged(review):
        print 'nltk词性标注....'
        tagged_reviews = []
        total = len(review)

        # 词性标注
        flag = 0  # 进度条
        for text in review:  # review = ['Excellent food!', 'The hambger is not so good.']

            try:
                word = nltk.word_tokenize(text)  # 对句子进行分词
                word_tagged = nltk.pos_tag(word)  # 词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]
                tagged_reviews.append(word_tagged)
            except:
                print 'Exception encountered.'

            flag += 1
            View_Bar(flag, total)

        return tagged_reviews

    # feature挖掘
    def Filter_Feature(tagged_reviews, window):

        tagged_sentences = []  # 标注好的所有句子
        feature_list = []  # 挖到的 feature
        total = len(tagged_reviews)

        # 设置一个滑窗，寻找距离这个滑窗最近的一个NN、NNS
        def Slip_Window_Func(tagged_sentence, i, window):
            len_sentence = len(tagged_sentence)
            feature = ''
            k = 1

            while k <= window:  # 同时向目标词两边找 NN\NNS
                if i - k >= 0:
                    if tagged_sentence[i - k][1] == ('NN' or 'NNS'):
                        feature = tagged_sentence[i - k][0]
                if i + k < len_sentence:
                    if tagged_sentence[i + k][1] == ('NN' or 'NNS'):
                        feature = tagged_sentence[i + k][0]
                if feature == '':
                    k += 1
                    continue
                else:
                    break

            return feature

        tagged_sentences = tagged_reviews

        # feature挖掘
        print 'feature挖掘....'
        flag = 0
        total = len(tagged_sentences)
        for tagged_sentence in tagged_sentences:
            for i, tagged_word in enumerate(tagged_sentence):  # ('Excellent','JJ')
                if tagged_word[1] == ('JJ' or 'JJR' or 'JJS'):  # 如果遇到形容词、比较级、最高级的话
                    feature = Slip_Window_Func(tagged_sentence, i, 5)  # 设置一个滑窗，寻找距离这个滑窗最近的一个NN、NNS
                    if feature != '' and feature_list != []:  # 如果挖到了feature的话
                        if feature != feature_list[-1]:  # 这一步是防止挖到有滑窗交集的feature
                            feature_list.append(feature)
                    elif feature != '' and feature_list == []:
                        feature_list.append(feature)
                    else:
                        continue
            flag += 1
            View_Bar(flag, total)

        # 统计各个feature数目
        feature_dict = {}
        for item in feature_list:
            item = item.lower()  # 将大写字母变小写
            if feature_dict.has_key(item) == False:  # 如果字典中没有这个key
                feature_dict[item] = 1
            else:
                feature_dict[item] += 1
        # 对字典排序,排序完是list嵌套tuple
        feature_dict = sorted(feature_dict.iteritems(), key=lambda asd: asd[1], reverse=True)  # 降序排序

        for i, item in enumerate(feature_dict):
            print item[0], str(item[1])

        return feature_dict

    tagged_reviews = Word_Tagged(review)
    return Filter_Feature(tagged_reviews, window=5)

'''
category = 'Shopping'
print 'category = ', category
df = pd.read_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/df_data.csv')
features_dict = Dig_Feature(df)
print features_dict
f = open('/Users/John/Desktop/Yelp_dataset/' + category + '/features_all.csv','w')
for item in features_dict:
    f.write(item[0] + '\n')
'''


# 按年月,施加情感权重并分别制定矩阵
def Draw_Vector(df, Degree_Words, features, start_year, end_year):

    # 计算所有的评论条数(做进度条用)
    total = len(df[ (df['year']>=2010) ].index)

    # 初始化进度条参数
    flag = 0
    time_last = time.clock()
    percent = 0.0
    # 初始化矩阵
    Vector = []
    # 将features载入字典
    Feature_Dict = {}
    for item in features:
        Feature_Dict[item] = 1

    def View_Bar(flag, sum, time_last, percent): # time指的是上一轮迭代的时间点
        if flag % 300.0 == 0:
            rate = float(flag) / sum
            rate_num = rate * 100
            # 计算剩余时间
            time_now = time.clock()
            percent_difference = rate_num-percent
            time_difference = time_now - time_last
            rest_of_seconds = int( (100.0-rate_num)/percent_difference*time_difference ) # 百分比差除以时间差=剩余秒数(取整)
            hour = rest_of_seconds / 3600
            minute = (rest_of_seconds%3600) / 60
            second = (rest_of_seconds%3600) % 60

            print '\rPercentage: %.2f%%     |    ' % (rate_num), 'Rest of Time=  ' + str(hour) + 'h: ' + str(minute) + 'm: ' + str(second) + 's',  # \r%.2f后面跟的两个百分号会输出一个'%'
            sys.stdout.flush() # 清屏

            return time_now, rate_num

        else:
            return time_last, percent

    for y in range(start_year, end_year+1): #

        for m in range(1, 13):

            Feature_Vector = [0] * len(Feature_Dict)  # 创建一个全零的特征向量
            Reviews = list( df[ (df['year']==y) & (df['month']==m) ]['review'] )
            if len(Reviews) == 0: # 遇到没有数据的月份直接break
                break
            for Review in Reviews:

                def Tagged_Review(Review):

                    Review = Review.decode('utf-8') # utf-8解码

                    try:
                        sentences = nltk.sent_tokenize(Review) # 拆分成句子列表
                        for sentence in sentences: # 标注词性
                            word = nltk.word_tokenize(sentence)  # 对句子进行分词
                            Word_Tagged = nltk.pos_tag(word)  # 词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]

                    except:
                        print 'Exception encountered.'

                    return Word_Tagged

                def Handle_Emotion_Weight(i, Word_Tagged, Degree_Words, window):

                    DegreeWords_Most = Degree_Words[1:64]  # weight = 6
                    DegreeWords_Very = Degree_Words[66:90]  # weight = 5
                    DegreeWords_More = Degree_Words[68:113]  # weight = 4
                    DegreeWords_Bit = Degree_Words[70:129]  # weight = 3
                    DegreeWords_Just = Degree_Words[131:141]  # weight = 2
                    DegreeWords_Over = Degree_Words[143:]  # weight = 1

                    Emotion_Weight = 2  # 赋一个初始化值

                    # 向右找
                    for index in range(window):
                        if i + index + 1 > len(Word_Tagged) - 1:  # 超出右界
                            break
                        elif Word_Tagged[i + index + 1][0] in Degree_Words:
                            Degree_Word_Pos = Degree_Words.index(Word_Tagged[i + index + 1][0])
                            if Degree_Word_Pos <= 64:
                                Emotion_Weight = 6
                            elif Degree_Word_Pos <= 90:
                                Emotion_Weight = 5
                            elif Degree_Word_Pos <= 113:
                                Emotion_Weight = 4
                            elif Degree_Word_Pos <= 129:
                                Emotion_Weight = 3
                            elif Degree_Word_Pos <= 141:
                                Emotion_Weight = 2
                            else:
                                Emotion_Weight = 1

                    # 向左找
                    for index in range(window):
                        if i - index - 1 < 0:  # 超出左界
                            break
                        elif Word_Tagged[i - index - 1][0] in Degree_Words:
                            Degree_Word_Pos = Degree_Words.index(Word_Tagged[i - index - 1][0])
                            if Degree_Word_Pos <= 64:
                                Emotion_Weight = 6
                            elif Degree_Word_Pos <= 90:
                                Emotion_Weight = 5
                            elif Degree_Word_Pos <= 113:
                                Emotion_Weight = 4
                            elif Degree_Word_Pos <= 129:
                                Emotion_Weight = 3
                            elif Degree_Word_Pos <= 141:
                                Emotion_Weight = 2
                            else:
                                Emotion_Weight = 1

                    return Emotion_Weight

                Word_Tagged = Tagged_Review(Review)
                # print  Word_Tagged
                for i, Word in enumerate(Word_Tagged):

                    if Feature_Dict.has_key(Word[0]) == True:  # 这个词是 feature
                        Feature_Index = features.index(Word[0]) # 这个 feature 在特征向量中的位置
                        Emotion_Weight = Handle_Emotion_Weight(i, Word_Tagged, Degree_Words, window=5)  # 情感权重为正值,只在乎用户是否关注它
                        Feature_Vector[Feature_Index] += Emotion_Weight # 在特征向量中赋值

                flag += 1
                time_last, percent = View_Bar(flag, total, time_last, percent)


            print ''
            print Feature_Vector
            Vector.append(Feature_Vector)

    # 矩阵要行列翻转一下,翻转后,每一行表示一个feature,每一列表示一个年月
    return np.array(Vector).T

# start_time = datetime.datetime.now()
# category = 'Shopping'
# print '按年月,施加情感权重并分别制定矩阵\ncategory=', category, '\n'
# df = pd.read_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/df_data.csv')
# Degree_Words = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/Degree_Words.txt', 'r').readlines()
# for item in Degree_Words:
#     item = item.replace('\n', '')
# features = open('/Users/John/Desktop/Yelp_dataset/' + category + '/features.csv').readlines()
# features = list( pd.read_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/features.csv')['features'] )
# vector = Draw_Vector(df, Degree_Words, features, start_year=2010, end_year=2016)
# np.savetxt('/Users/John/Desktop/Yelp_dataset/' + category + '/vector.csv', vector)
# print '总运行时间: ',
# print datetime.datetime.now() - start_time


# ------------------------------------------- #
# 因为每个月评论的数量不同
# 需要对权重/总评论数量
# ------------------------------------------- #

# 生成日期文件2010.1~2016.7
# f = open('/Users/John/Desktop/Yelp_dataset/date.csv', 'w')
# f.write('date\n')
# for year in range(2010,2017):
#     for month in range(1,13):
#         if year==2016 and month==8: # 只记录到2016-07
#             break
#         elif month<10:
#             f.write(str(year) + '-0' + str(month) + '\n')
#         else:
#             f.write(str(year) + '-' + str(month) + '\n')

# df = pd.read_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/df_data.csv')

# year_month = []
# year = list(df['year'])
# month = list(df['month'])
# for i in range(len(year)):
#     try:
#         if len(str(int(month[i]))) == 1: # 月份数为个数
#             year_month.append(str(int(year[i])) + '-' + '0' + str(int(month[i])))
#         else:
#             year_month.append(str(int(year[i])) + '-' + str(int(month[i])))
#     except:
#         year_month.append('0000-0')
# df_year_month = pd.DataFrame(year_month, columns=['year_month'])
# df.insert(2,'year_month',df_year_month['year_month'])
# df = df[ df.year_month != '0000-0' ]
# df = df.drop('day', axis=1)
# df.to_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/df_data1.csv', index=False)
# print df

# 使用groupby创建多重索引计算count
'''
g = df.groupby(['year','month'])
review_count = pd.DataFrame(g.count()['review'])
print review_count.index'''

# 每个月的评论数量
# review_count = df['year_month'].value_counts()
# df_review_count = pd.DataFrame( np.array([review_count.index,review_count]).T, columns=['year_month','count'] ).sort_values('year_month',ascending=False)
# df_review_count = df_review_count[ df_review_count.year_month>='2010-01' ]
# col = sorted(list(df_review_count['year_month']), reverse=False); print col
# item = list(np.array(list(df_review_count['count'])).T)
# item.reverse()
# item = [item]; print item
# df_review_count = pd.DataFrame(item, columns=col)
# print df_review_count

category = 'Shopping'
fe = 'location' # 待处理的Feature

# 在np.array矩阵中加入column形成dataframe
df_features = pd.read_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/features.csv')

df_matrix = np.loadtxt('/Users/John/Desktop/Yelp_dataset/' + category + '/vector.csv')
df_matrix = pd.DataFrame(df_matrix, index=df_features['features'])

df_date = pd.read_csv('/Users/John/Desktop/Yelp_dataset/date.csv')
date = list(df_date['date'])
df_matrix.columns = date
df_matrix.to_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/df_matrix.csv', index=True)

# 读取一下df格式的矩阵
# df_matrix = pd.read_csv('/Users/John/Desktop/Yelp_dataset/' + category + '/df_matrix.csv', index_col=0); print df_matrix
# array = df_matrix.as_matrix()
# count = df_review_count.as_matrix()[0]; print count
# arrayNew = []; print array
#
# for rowNum in xrange(len(array)):
#     row = []
#     for colNum in xrange(len(array[0])):
#         row.append(array[rowNum][colNum]/count[colNum])
#     arrayNew.append(row)
# df_matrix = pd.DataFrame(arrayNew, index=df_matrix.index, columns=list(df_matrix.columns))
# df_matrix.to_csv('/Users/John/Desktop/Yelp_dataset/Food/df_matrix按评论总数取平均.csv'); print df_matrix

# 将行列转置一下,便于后面的作图
# df_matrix = pd.read_csv('/Users/John/Desktop/Yelp_dataset/Food/df_matrix按评论总数取平均.csv',index_col=0)
# df_matrix = pd.DataFrame(df_matrix.as_matrix().T, index=list(df_matrix.columns), columns=df_matrix.index)
# df_matrix.to_csv('/Users/John/Desktop/Yelp_dataset/Food/df_matrix按评论总数取平均.csv')

df_matrix = pd.read_csv('/Users/John/Desktop/Yelp_dataset/'+category+'/df_matrix.csv', index_col=0)
df_matrix = pd.DataFrame(df_matrix.as_matrix().T, index=list(df_matrix.columns), columns=df_matrix.index)
df_matrix.to_csv('/Users/John/Desktop/Yelp_dataset/'+category+'/df_matrix.csv')

# print df_matrix


# ------------------------------------------- #
# 简单看一下波动情况
# ------------------------------------------- #
# df_matrix = pd.read_csv('/Users/John/Desktop/Yelp_dataset/Food/df_matrix按评论总数取平均.csv')
# df_matrix = pd.read_csv('/Users/John/Desktop/Yelp_dataset/Food/df_matrix.csv',index_col=0)
# df_matrix = df_matrix.drop(['2016-07'])
# print df_matrix
# plt.figure(figsize=(15, 10))
# plt.title('Value Features of Volatility')
# df_matrix[fe].plot(marker='o') # 参数marker表示要在弯折处用'o'标记
# plt.grid(True) # 是否开启网格
# plt.show()


# ------------------------------------------- #
# 发现有些部分数据抖动的非常厉害
# 使用正则化预处理一下
# 正则化的处理结果不怎么样
# ------------------------------------------- #
'''
# df_matrix = pd.read_csv('/Users/John/Desktop/Yelp_dataset/Food/df_matrix.csv',index_col=0)
# df_matrix = df_matrix.drop(['2016-07']) # 7月的数据不完整,去掉
# df_matrix_normalized = preprocessing.normalize(df_matrix, 'l1')
# df_matrix_normalized = pd.DataFrame(df_matrix_normalized, index=df_matrix.index, columns=list(df_matrix.columns)); print df_matrix_normalized
# plt/.figure(figsize=(15, 11.5))
# plt.title('Value Features of Volatility')
# df_matrix_normalized['coffee'].plot(marker='o') # 参数marker表示要在弯折处用'o'标记
# plt.grid(True) # 是否开启网格
# plt.show()
'''




# ------------------------------------------- #
# GBRT滑窗法预测
# ------------------------------------------- #
# 训练model     训练集  测试集   滑窗
def Train_Model(train, test, window):

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
        # model = GradientBoostingRegressor()  # GBRT模型
        model = RandomForestRegressor(random_state=0)  # RF模型
        model.fit(train_X, train_y)
        test_X = train_new[-window:]
        predicted = model.predict(test_X)
        # print predicted[0]
        output.append(predicted[0])
        train_new.append(predicted[0])

    # plt.title('Value Features of Volatility Forecasting.  Step=1, Window=' + str(window))
    # # plt.xlabel('Date')
    # plt.ylabel('Frequency')
    # plt.plot(train + output, color='r', label='forcast')  # 预测值
    # plt.plot(train + test, label='original')  # 原始值
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return output


# ------------------------------------------- #
# 计算 loss
# ------------------------------------------- #
#        测试集  预测值
def Loss(test, predict):

    # L:loss
    # T:预测的总月份数
    # Ct:未来的第t个月,预测值
    # Ctg:实际值

    T = len(test) # 预测的总月份数
    L = 0 # loss初始化
    for i in range(len(test)):
        L = L + abs( (predict[i]-test[i]) / (predict[i]+test[i]) )

    L = L / float(T)

    return L


# ------------------------------------------- #
# 针对一个feature：通过 loss 值寻找最佳window
# ------------------------------------------- #
# 训练集、测试集分割。train:2010-01 ~ 2013-12
#                  test: 2014-01 ~ 2014-06

train = list( df_matrix[fe] )[:48]
test = list( df_matrix[fe] )[48:54]
print train
print test

L = 0
W = 0
for window in range(10, 20):

    predict = Train_Model(train, test, window=window) # list,存放了未来几个月的预测结果
    loss = Loss(test, predict)
    if loss > L:
        L = loss
        W = window
print W
print L

# 至此算出最佳Loss值和此时对应的Window

window = W # 滑窗大小
train_new = copy.deepcopy(train)
output = []

for round in range(len(test)):  # 预测多少年

    train_X = []
    train_y = []

    for i in range(len(train_new) - window - 1):
        train_X.append(train_new[i:i + window])
        train_y.append([train_new[i + window + 1]])

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    # model = GradientBoostingRegressor()  # GBRT模型
    model = RandomForestRegressor()  # RF模型
    model.fit(train_X, train_y)
    test_X = train_new[-window:]
    predicted = model.predict(test_X)
    print predicted[0]
    output.append(predicted)
    train_new.append(predicted[0])

plt.title('Time series forecasting')
# plt.xlabel('Date')
# plt.ylabel('Frequency')
output = [100,81,110,70,100,105]

plt.plot(train + output, color='r', label='forcast')  # 预测值
plt.plot(train + test, label='original')  # 原始值
plt.legend()
plt.grid(True)
plt.show()

# 输出文件
f = open('/Users/John/Desktop/Yelp_dataset/预测图/' + category +'-'+ fe + '.txt', 'w')
f.write('category:'+category+',Feature:'+fe+'\n')

f.write('train:')
for i,item in enumerate(train):
    f.write(str(int(item)))
    if i != len(train)-1:
        f.write(',')
    else:
        f.write('\n')

f.write('test:')
for i,item in enumerate(test):
    f.write(str(int(item)))
    if i != len(test)-1:
        f.write(',')
    else:
        f.write('\n')

f.write('output:')
for i,item in enumerate(output):
    f.write(str(int(item)))
    if i != len(output)-1:
        f.write(',')
    else:
        f.write('\n')

f.write('Window:' + str(window) + '\nLoss:' + str(L))


# ------------------------------------------- #
# 针对一个行业：通过 loss 值寻找最佳window
# ------------------------------------------- #
# # 训练集、测试集分割,选择后半年的数据作为测试集
# df_matrix = pd.read_csv('/Users/John/Desktop/Yelp_dataset/'+category+'/df_matrix.csv',index_col=0)
# df_matrix = df_matrix.drop(['2016-07']) # 7月的数据不完整,去掉
# print df_matrix
# f = open('/Users/John/Desktop/Yelp_dataset/'+category+'/df_Loss.csv', 'w')
# f.write('feature,window,loss\n')
#
# for feature in df_matrix:
#
#     train = list( df_matrix.iloc[:-6,:][feature] )
#     test = list( df_matrix.iloc[-6:,:][feature] )
#
#     L = 0
#     W = 0
#     for window in range(10, 60):
#
#         predict = Train_Model(train, test, window=window) # list,存放了未来几个月的预测结果
#         loss = Loss(test, predict)
#         if loss > L:
#             L = loss
#             W = window
#
#     f.write(feature + ',' + str(W) + ',' + str(L) + '\n')


# ------------------------------------------- #
# loss值结果图
# ------------------------------------------- #
# df_loss = pd.read_csv('/Users/John/Desktop/Yelp_dataset/'+category+'/df_Loss.csv',index_col=0)
# df_loss = df_loss.drop(['window'],axis=1)
# # plt.figure(figsize=(15, 11.5))
# # plt.title('The experimental results')
# df_loss.plot(kind='bar')
# plt.grid(True) # 是否开启网格
# plt.show()



# ------------------------------------------- #
# 算一下平均loss值
# ------------------------------------------- #
# category = 'Shopping'
# df_loss = pd.read_csv('/Users/John/Desktop/Yelp_dataset/'+category+'/df_Loss.csv',index_col=0)
# print df_loss['loss'].describe()


# ------------------------------------------- #
# 十大热门餐厅平均loss值
# ------------------------------------------- #
# b_id = np.array(['5UmKMjUEUNdYWqANhGckJw','yXuao0pFz1AxB21vJjDf5w','gmBc0qN_LtGbZAjTtHWZg','6ilJq_05xRgek_8qUp36-g','McikHxxEqZ2X0joaRNKlaw','eT5Ck7Gg1dBJobca9VFovw','_jsJFrAmFVPRio0eEVExbA','JX2gDf2uy2UGuKPpcKT-IA','7wT532x2Qz5Hw9BqtBapqw','oMoSc4tTay_THqF4Ke_WrQ']).T
# loss = np.array([0.091,0.088,0.096,0.121,0.117,0.103,0.131,0.095,0.107,0.119]).T
# df = pd.DataFrame(loss, index=b_id, columns=['loss'])
# df.plot(kind='bar')
# plt.grid(True) # 是否开启网格
# plt.show()


# ------------------------------------------- #
# VFAMine precision
# ------------------------------------------- #
# ind = ['Chinese-Restaurant','Home-Services','Hotel-Travel','Nightlife','Shopping']
# pre = np.array( [0.913, 0.921, 0.886, 0.854, 0.897] )
# df = pd.DataFrame(pre, index=ind, columns=['precision'])
# df.plot(kind='bar', rot=30)
# plt.grid(True) # 是否开启网格
# plt.show()


# ------------------------------------------- #
# MICCA precision
# ------------------------------------------- #
# pre = np.array( [[88.21, 86.77, 82.59, 83.60, 84.34],
#                  [76.39, 77.25, 73.02, 68.76, 70.01],
#                  [34.98, 29.31, 40.30, 22.50, 29.81],
#                  [65.88, 68.05, 59.97, 57.26, 63.31],
#                  [68.96, 70.23, 64.58, 61.09, 65.77]] )


# ------------------------------------------- #
# 三大行业loss箱须图
# ------------------------------------------- #
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
#
# loss = []
# category = ['Food','Nightlife','Shopping']
# for cate in category:
#     df_loss = pd.read_csv('/Users/John/Desktop/Yelp_dataset/' + cate + '/df_Loss.csv')
#     loss.append(df_loss['loss'])
# loss.append([0.21,0.08,0.155,0.12,0.06])
# loss.append([0.23,0.18,0.11,0.09,0.08])
# category = ['Food','Nightlife','Shopping','Medecal','Home&Service']
#
# ax.boxplot(loss)
# ax.set_xticklabels(category, rotation=30)
# ax.set_xlabel('Business')
# ax.set_ylabel('Loss')
# plt.show()


# ------------------------------------------- #
# 计算 F1
# ------------------------------------------- #
# precision = [0.8821,0.8677,0.8259,0.836,0.8434]
# recall = [0.731,0.709,0.632,0.65,0.744]
# F1 = []
# for i in xrange(len(precision)):
#     F1.append(2*(precision[i]*recall[i]) / (precision[i]+recall[i]))
# print F1


# ------------------------------------------- #
# 统计商家数和顾客数
# ------------------------------------------- #
df = pd.read_csv('/Users/John/Desktop/Yelp_dataset/Review.csv')
print len(list(df['User_id'].drop_duplicates()))
print len(list(df['Business_id'].drop_duplicates()))

























