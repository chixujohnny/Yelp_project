# coding: utf-8

import pandas as pd
import json
import matplotlib.pyplot as plt
import sys
import nltk
import chardet

import Yelp_Filter_business
import Yelp_Filter_review
import Yelp_WordTag_review
import Yelp_FeatureDig_review
import Yelp_EmotionProcess


##############################
#  更新数据结构(Dataframe格式)  #
##############################
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

df = pd.read_csv('/Users/John/Desktop/Yelp_dataset/Food/df_data.csv')
features_dict = Dig_Feature(df)
print features_dict
f = open('/Users/John/Desktop/Yelp_dataset/Food/top_50_features.csv','w')
for item in features_dict:
    f.write(item[0] + '\n')




















