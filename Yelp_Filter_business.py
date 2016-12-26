# coding:utf-8

######################################
#  通过商家类别筛选出所有的 business_id  #
######################################

import json

#  保存所有的服务类别
def All_Business_Category(business_path):
    print '保存所有的服务类别....',

    categories_dict = {}
    lines = open(business_path, 'r').readlines()
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/all_business_category.csv', 'w')

    for line in lines:
        line_json = json.loads(line) # 字符串 -> json格式
        categories = line_json["categories"]
        for category in categories:
            if categories_dict.has_key(category) == 0: # 如果category字典没这个key
                categories_dict[category] = 1
            else: # 有这个key
                categories_dict[category] += 1

    # 对字典进行排序 [(u'Restaurants', 26729), (u'Shopping', 12444)]
    categories_dict = sorted(categories_dict.iteritems(), key=lambda asd:asd[1], reverse=True)

    for category in categories_dict: # category = u'Books, Mags, Music & Video'
        f.write(str(category[0]) + ',' + str(category[1]) + '\n') # 保存一下category 以及频度

    print 'Done!'


#  筛选指定的category
def Filter_Business_Category(business_path, filter_category):
    print '筛选指定的category....'

    filter_business_id = []
    lines = open(business_path, 'r').readlines()
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Bid.csv', 'w')

    for line in lines:
        flag = True
        line_json = json.loads(line) # 字符串 -> json格式
        categories = line_json["categories"] # ['Fastfood', 'Restaurant']
        business_id = line_json["business_id"] # "5UmKMjUEUNdYWqANhGckJw"
        for item in filter_category:
            if item not in categories:
                flag = False
                break
        if flag == True: # 表示该business满足filter_category
            filter_business_id.append(business_id)
        else: # 不满足filter条件,跳到下一个商户
            continue

    # 写文件
    for item in filter_business_id:
        f.write(item + '\n')


if __name__ == '__main__':
    business_path = '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'

    # 统计所有的 business_category
    # All_Business_Category(business_path)

    # 筛选指定category的business
    filter_category = ['Nightlife'] # 要保证所有的类别在business中均有体现
    Filter_Business_Category(business_path, filter_category)



