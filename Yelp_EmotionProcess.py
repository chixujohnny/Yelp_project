    # coding: utf-8

################################
#  分析 review 文件并施加情感权重  #
################################

import os, sys, time
import chardet
import nltk

#
#  进度条
#
def View_Bar(flag, sum):
    rate = float(flag) / sum
    rate_num = rate * 100
    if flag % 15.0 == 0:
        print '\r%.2f%%: ' % (rate_num),  # \r%.2f后面跟的两个百分号会输出一个'%'
        sys.stdout.flush()


#
#  review 预处理, 生成两个字典, UserID_review_dict, BusinessID_review_dict
#
def Preprocess_Review(review_path): # review 的文件路径
    print 'review预处理....'

    UserID_review_dict = {}
    BusinessID_review_dict = {}
    lines = open(review_path, 'r').readlines()

    for line in lines:
        User_id = line[:22]
        Business_id = line[23:45]
        Review_text = line[46:]

        # UserID_review_dict
        if UserID_review_dict.has_key(User_id) == False:
            UserID_review_dict[User_id] = [Review_text]
        else:
            UserID_review_dict[User_id].append(Review_text)

        # BusinessID_review_dict
        if BusinessID_review_dict.has_key(Business_id) == False:
            BusinessID_review_dict[Business_id] = [Review_text]
        else:
            BusinessID_review_dict[Business_id].append(Review_text)

    return UserID_review_dict, BusinessID_review_dict

#
#  施加情感权重,并制作矩阵
#
def Process_Emotion_Weight(Feature, Degree_Words, UserID_review_dict, BusinessID_review_dict, uid_vector_path, bid_vector_path):
    print '施加情感权重....'

    Feature_Dict = {}
    for item in Feature:
        Feature_Dict[item] = 1

    # 处理 User_Review 并标注词性
    def Tagged_Review(Review):

        # utf-8解码
        Review = Review.decode('utf-8')

        # 拆分成句子列表
        sentences = nltk.sent_tokenize(Review)

        # 标注词性
        for sentence in sentences:
            word = nltk.word_tokenize(sentence) # 对句子进行分词
            Word_Tagged = nltk.pos_tag(word) # 词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]

        return Word_Tagged

    # 处理 Emotion_Weight
    def Handle_Emotion_Weight(i, Word_Tagged, Degree_Words, window):

        DegreeWords_Most = Degree_Words[1:64]    # weight = 6
        DegreeWords_Very = Degree_Words[66:90]   # weight = 5
        DegreeWords_More = Degree_Words[68:113]  # weight = 4
        DegreeWords_Bit = Degree_Words[70:129]   # weight = 3
        DegreeWords_Just = Degree_Words[131:141] # weight = 2
        DegreeWords_Over = Degree_Words[143:]    # weight = 1

        Emotion_Weight = 2 # 赋一个初始化值

        # 向右找
        for index in range(window):
            if i+index+1 > len(Word_Tagged)-1: # 超出右界
                break
            elif Word_Tagged[i+index+1][0] in Degree_Words:
                Degree_Word_Pos = Degree_Words.index(Word_Tagged[i+index+1][0])
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
            if i-index-1 < 0: # 超出左界
                break
            elif Word_Tagged[i-index-1][0] in Degree_Words:
                Degree_Word_Pos = Degree_Words.index(Word_Tagged[i-index-1][0])
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

    # 处理 UserID_review_dict
    print '处理UserID_review_dict'
    f = open(uid_vector_path, 'w')
    flag = 0
    total = len(UserID_review_dict)
    All_User_Feature_Vector = []

    # 先写入 columns
    Feature_Columns = ''
    for i, item in enumerate(Feature):
        Feature_Columns += item
        if i != len(Feature)-1:
            Feature_Columns += ','

    f.write('Business_ID,' + Feature_Columns + '\n')

    for UserID in UserID_review_dict:
        Feature_Vector = [0] *len(Feature_Dict) # 创建一个全零的特征向量
        User_Review = UserID_review_dict[UserID]

        for Review in User_Review:
            Word_Tagged = Tagged_Review(Review) # [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]

            for i, Word in enumerate(Word_Tagged):
                if Feature_Dict.has_key(Word[0]) == True: # 这个词是 feature
                    Feature_Index = Feature.index(Word[0])
                    Emotion_Weight = Handle_Emotion_Weight(i, Word_Tagged, Degree_Words, window=5) # 情感权重为正值,只在乎用户是否关注它
                    Feature_Vector[Feature_Index] += Emotion_Weight
        All_User_Feature_Vector.append([UserID, Feature_Vector])
        # 写文件
        f.write(UserID + ', ' + str(Feature_Vector)[1:-1] + '\n')
        flag += 1
        View_Bar(flag, total)
    print 'Done!'

    # 处理 Business_review_dict
    print '处理Business_review_dict'
    f = open(bid_vector_path, 'w')
    flag = 0
    total = len(BusinessID_review_dict)
    All_Business_Feature_Vector = []

    # 先写入 columns
    Feature_Columns = ''
    for i, item in enumerate(Feature):
        Feature_Columns += item
        if i != len(Feature)-1:
            Feature_Columns += ','

    f.write('Business_ID,' + Feature_Columns + '\n')

    for BusinessID in BusinessID_review_dict:
        Feature_Vector = [0] * len(Feature_Dict)  # 创建一个全零的特征向量
        Business_Review = BusinessID_review_dict[BusinessID]

        for Review in Business_Review:
            Word_Tagged = Tagged_Review(Review)

            for i, Word in enumerate(Word_Tagged):
                if Feature_Dict.has_key(Word[0]) == True:
                    Feature_Index = Feature.index(Word[0])
                    Emotion_Weight = Handle_Emotion_Weight(i, Word_Tagged, Degree_Words, window=5)  # 情感权重为正值,只在乎用户是否关注它
                    Feature_Vector[Feature_Index] += Emotion_Weight
        All_Business_Feature_Vector.append([BusinessID, Feature_Vector])
        # 写文件
        f.write(BusinessID + ', ' + str(Feature_Vector)[1:-1] + '\n')
        flag += 1
        View_Bar(flag, total)

    print 'Done!'


#        #
#  main  #
#        #

# Feature = []
# lines = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Feature.txt', 'r').readlines()
# for i, item in enumerate(lines):
#     Feature.append(item.replace('\n', '').split(',')[0])
#     if i == 499: # 只取前500个feature
#         break
#
# Degree_Words = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/Degree_Words.txt', 'r').readlines()
# for item in Degree_Words:
#     item = item.replace('\n', '')
#
# UserID_review_dict, BusinessID_review_dict = Preprocess_Review('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Review.txt')
#
# Process_Emotion_Weight(Feature, Degree_Words, UserID_review_dict, BusinessID_review_dict, vector_path='')




