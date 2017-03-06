# coding: utf-8

#####################
#  基于协同过滤的推荐  #
#####################


import json


def CF_Data_Preprocess(review_path):

    #  搞一个 dict
    #  {'user_id':['business_id', 'stars', 'date']}

    User_Rate_Dict = {}

    lines = open(review_path)
    for line in lines:
        line_json = json.loads(line)
        uid = line_json['user_id']
        bid = line_json['business_id']
        stars = line_json['stars']
        date = line_json['date']




