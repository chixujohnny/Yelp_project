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
                for i, Word in enumerate(Word_Tagged):

                    if Feature_Dict.has_key(Word[0]) == True:  # 这个词是 feature
                        Feature_Index = features.index(Word[0]) # 这个 feature 在特征向量中的位置
                        Emotion_Weight = Handle_Emotion_Weight(i, Word_Tagged, Degree_Words, window=5)  # 情感权重为正值,只在乎用户是否关注它
                        Feature_Vector[Feature_Index] += Emotion_Weight # 在特征向量中赋值

                flag += 1
                time_last, percent = View_Bar(flag, total, time_last, percent)

            Vector.append(Feature_Vector)

    # 矩阵要行列翻转一下,翻转后,每一行表示一个feature,每一列表示一个年月
    return np.array(Vector).T

# start_time = datetime.datetime.now()
# category = 'Nightlife'
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


# 看一下feature的波动情况


# 将矩阵保存成 Dataframe 格式
def Vector_to_Dataframe(vector, feature):


























