# coding: utf-8

import pandas as pd
import json

import Yelp_Filter_business
import Yelp_Filter_review
import Yelp_WordTag_review
import Yelp_FeatureDig_review
import Yelp_EmotionProcess


###########################
#  更新数据结构(Dataframe格式)  #
###########################
def Review_DataStructure_Pd(yelp_review_path, yelp_business_path, review_csv):

    lines = open(yelp_review_path, 'r').readlines()
    f = open(review_csv, 'w')

    for line in lines:
        line_json = json.loads(line)
        uid = line_json['user_id'].encode('utf-8')
        bid = line_json['business_id'].encode('utf-8')
        year = line_json['date'].split('-')[0].encode('utf-8')
        month = line_json['date'].split('-')[1].encode('utf-8')
        day = line_json['date'].split('-')[2].encode('utf-8')
        review = '"' + line_json['text'].encode('utf-8').replace(('\n' or '\n\n' or ','), '.').replace('"', "'") + '"'
        f.write(uid + ',' + bid + ',' + year + ',' + month + ',' + day + ',' + review + '\n')

    return 0

def GroupBy(review_csv):

    data = pd.read_csv(review_csv)
    data = data.groupby(['year','month'])
    df = pd.DataFrame(data.count()['User_id'])
    print df

yelp_review_path = '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
save_path = '/Users/John/Desktop/Yelp_dataset/Review.csv'
# Review_DataStructure_Pd(yelp_review_path, save_path)
# GroupBy(save_path)


######################################
#  通过商家类别筛选出所有的 business_id  #
######################################
# business_path = '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
# filter_category = ['Restaurants']  # 要保证所有的类别在business中均有体现
# filter_category_string = ''
# for i, item in enumerate(filter_category):
#     if i != len(filter_category)-1:
#         filter_category_string += item + '_'
#     else:
#         filter_category_string += item
# business_id = Yelp_Filter_business.Filter_Business_Category(business_path, filter_category)


################################
#  通过商家id筛选出所有的 review  #
################################
# review_path = '/Users/John/Desktop/' + filter_category_string + '_Review.txt'
# review = Yelp_Filter_review.Filter_Review(business_id, review_path)