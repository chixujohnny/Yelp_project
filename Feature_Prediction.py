# coding: utf-8

import pandas as pd
import json
import matplotlib.pyplot as plt

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
df = pd.read_csv(review_csv)
df_Restaurants = df[df['category'].str.contains('Restaurants',na=False)]
df_Nightlife = df[df['category'].str.contains('Nightlife',na=False)]
df_Shopping = df[df['category'].str.contains('Shopping',na=False)]
df_ChineseRestaurants = df[df['category'].str.contains(['Chinese','Restaurants'],na=False)]
print df_ChineseRestaurants
















