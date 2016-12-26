# coding:utf-8

################################
#  通过商家id筛选出所有的 review  #
################################

import json

#  通过business_id提取review
def Filter_Review(business_id): # business_id = ['UsFtqoBl7naz8AVUBZMjQQ', 'mVHrayjG3uZ_RLHkLj-AMg']
    print '根据business_id筛选评论信息....',

    business_id_dict = {}
    lines = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json', 'r').readlines()
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Review.txt', 'w')

    # 创建一个字典存business_id (为了让查找速度更快)
    for item in business_id:
        business_id_dict[item] = 1

    for line_num, line in enumerate(lines):
        line_json = json.loads(line) # 字符串 -> json格式
        bid = line_json['business_id']
        uid = line_json['user_id']
        if business_id_dict.has_key(bid) == True: # 是要找的business
            review = line_json['text'].replace('\n', '').replace('\n\n', '')
            f.write(str(uid.encode('utf-8')) + '-' + str(bid.encode('utf-8')) + ':' + str(review.encode('utf-8')) + '\n')

    print 'Done!'


if __name__ == '__main__':
    business_id = []
    lines = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Bid.csv', 'r').readlines()
    for line in lines:
        business_id.append(line[:-1]) # 要把最后的换行符去掉

    Filter_Review(business_id)