# coding:utf-8

import Yelp_Filter_business
import Yelp_Filter_review
import Yelp_WordTag_review
import Yelp_FeatureDig_review
import Yelp_EmotionProcess
import Yelp_DrawHeatmap2

# ######################################
# #  通过商家类别筛选出所有的 business_id  #
# ######################################
business_path = '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
filter_category = ['Restaurants', 'Chinese']  # 要保证所有的类别在business中均有体现
filter_category_string = ''
for i, item in enumerate(filter_category):
    if i != len(filter_category)-1:
        filter_category_string += item + '_'
    else:
        filter_category_string += item
# business_id = Yelp_Filter_business.Filter_Business_Category(business_path, filter_category)
#
#
#
# ################################
# #  通过商家id筛选出所有的 review  #
# ################################
review_path = '/Users/John/Desktop/' + filter_category_string + '_Review.txt'
# review = Yelp_Filter_review.Filter_Review(business_id, review_path)



#############################
#  对 review 内容进行词性标注  #
#############################
# tagged_reviews = Yelp_WordTag_review.Word_Tagged(review)



#####################################
#  对标注好词性的review进行feature抽取  #
#####################################
# f = open('/Users/John/Desktop/' + filter_category_string + '_Feature.txt', 'w')
# feature_dict = Yelp_FeatureDig_review.Filter_Feature(tagged_reviews, window=5)
# for i, item in enumerate(feature_dict):
#     if i == 500:
#         break
#     else:
#         f.write(item[0] + ',' + str(item[1]) + '\n')



################################
#  分析 review 文件并施加情感权重  #
################################
# 提取前500个feature
# feature = []
# lines = open('/Users/John/Desktop/' + filter_category_string + '_Feature.txt', 'r').readlines()
# for line in lines:
#     line = line.replace('\n', '').split(',')
#     feature.append(line[0])
#
#
# # 初始化程度词
# Degree_Words = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/Degree_Words.txt', 'r').readlines()
# for item in Degree_Words:
#     item = item.replace('\n', '')
#
# # 载入一下user的评论,和business下的评论
# UserID_review_dict, BusinessID_review_dict = Yelp_EmotionProcess.Preprocess_Review(review_path)
#
# # 情感分析
uid_vector_path = '/Users/John/Desktop/' + filter_category_string + '_User_Emotion_Vector.txt'
bid_vector_path = '/Users/John/Desktop/' + filter_category_string + '_Business_Emotion_Vector.txt'
# Yelp_EmotionProcess.Process_Emotion_Weight(feature, Degree_Words, UserID_review_dict, BusinessID_review_dict, uid_vector_path, bid_vector_path)



#############
#  绘制热图  #
############
data_user = Yelp_DrawHeatmap2.Heatmap_Data_Preprocess(uid_vector_path)
Yelp_DrawHeatmap2.Heatmap_Draw(data_user)

data_bid = Yelp_DrawHeatmap2.Heatmap_Data_Preprocess(bid_vector_path)
Yelp_DrawHeatmap2.Heatmap_Draw(data_bid)












