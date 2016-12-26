# coding:utf-8

import nltk, os, sys
from nltk import SnowballStemmer

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
#  筛选 指定类型的 店家
#
def Filter_Business(path, type):  # path是business表路径，type:待筛选店家的categories
    lines = open(path, 'r').readlines()
    len_lines = len(lines)
    filter_bussiness_id = {}  # 筛选完成的商家id的字典 {"5UmKMjUEUNdYWqANhGckJw":None, "UsFtqoBl7naz8AVUBZMjQQ":None, }

    flag = 0  # 进度条
    print '筛选制定类型店家:'
    for line in lines:
        line = line.split('"categories": ')
        categories = line[1].split(', "city"')[0]  # ["Fast Food", "Restaurants"]
        business_id = line[0].split(', "full_address"')[0].split('"business_id": ')[1]
        if type in categories:
            filter_bussiness_id[business_id[1:-1]] = 0
        flag += 1
        View_Bar(flag, len_lines)

    print ('字典: ' + str(filter_bussiness_id))
    print ('字典长度: ' + str(len(filter_bussiness_id)))

    return filter_bussiness_id

#
#  提取指定类型店家的 review
#
def Filter_Business_Review(path, filter_business_id):  # path是review文件路径
    lines = open(path, 'r').readlines()
    len_lines = len(lines)
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/review_text.txt', 'w')

    flag = 0  # 进度条
    print '提取指定店家的review:'
    for line in lines:
        dict = eval(line)
        business_id = dict['business_id']  # 该评论对应的商家
        if filter_business_id.has_key(business_id) == True:
            text = dict['text'].replace('\n', '')
            f.write(text + '\n')
        flag += 1
        View_Bar(flag, len_lines)

    return 0

#            #
#  词性标注  #
#            #
def Tag_Word(path):  # path 是所有用户的评论文件路径
    lines = open(path, 'r').readlines()
    len_lines = len(lines)
    tags = []  # 保存每个文章分词后的词性 [ [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')],
    #                           [('Superb', 'NNP'), ('customer', 'NN'), ('service', 'NN'), ('.', '.')] ]
    feature_word = []  # 提出的服务价值分布特征

    # 分词、赋词性
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/word_tagged_sentences.txt',
             'w')  # 保存一下词性标注后的结果
    flag = 0  # 进度条
    print '词性标注:'
    for text in lines:
        sentences = nltk.sent_tokenize(text)  # 将文本拆分成句子列表
        # 先对每个句子进行分词，在对这个句子进行词性标注（这样效果比较好）
        for sentence in sentences:
            word = nltk.word_tokenize(sentence)  # 先对句子进行分词 1
            word_tagged = nltk.pos_tag(word)  # 再对这个分好的句子进行词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]
            for item in word_tagged:  # 将标注好的词写入文件中
                f.write(item[0] + '/' + item[1] + ' ')  # 'Excellent/JJ food/NN ./. '
            f.write('\n')  # 这里我认为每个能展现feature的评论都是蕴含在一句话中的，因此每句话一行，到时候找feature的时候也是一行一行的去找
        flag += 1
        View_Bar(flag, len_lines)

    return 0

#                     #
#  筛选 feature 词汇  #
#                     #
def Featuer_Word(path, window):  # path 是词性标注后的评论句子
    flag = 0  # 进度条
    lines = open(path, 'r').readlines()
    len_lines = float(len(lines))
    tagged_sentences = []  # 保存所有标注好的句子
    # [ [(“'Excellent','JJ'), ('food','NN'), ('.','.')],
    #   [('Superb','NNP'), ('customer','NN'), ('service','NN'), ('.','.')] ]
    feature_list = []  # 挖到的feature

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

    # 数据预处理
    flag = 0  # 进度条
    print ('数据预处理进度: ')
    for line in lines:  # 预处理一下字符串 'Excellent/JJ food/NN ./. \n'
        sentence = line[:-3].split(' ')  # ['Excellent/JJ', 'food/NN', './.']
        tagged_sentence = []  # 标注好的一个句子 [('Excellent','JJ'), ('food','NN'), ('.','.')]
        for item in sentence:
            tagged_sentence.append(item.split('/'))
        tagged_sentences.append(tagged_sentence)
        flag += 1
        View_Bar(flag, len_lines)
        # if flag == 100:
        #     break
    print('')

    # 使用滑窗window确定 feature
    flag = 0  # 进度条
    print ('feature挖掘进度: ')
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
        View_Bar(flag, len_lines)

    # 将feature词汇保存一下
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/feature.txt', 'w')
    for item in feature_list:
        f.write(str(item) + '\n')
    print('feature词汇保存完毕')

#
#  对 feature 词汇进行再清洗(词干提取)
#
def Feature_Data_Cleaning(path, stemmer):  # path是装有feature词汇的文件路径, stemmer=True/False是否提取词干
    lines = open(path, 'r').readlines()
    len_lines = len(lines)
    feature_dict = {}  # 保存feature的字典

    # 初始化词干提取器
    stemmer = SnowballStemmer("english")

    # 把原始文件放到字典中
    flag = 0
    print '将feature整理成字典:'
    for feature in lines:
        feature = feature[:-1]
        if stemmer == True:
            feature = stemmer.stem(feature)
        if feature_dict.has_key(feature) == False:  # 如果字典里没有这个feature
            feature_dict[feature] = 1  # 赋一下key-value对
        else:  # 如果有这个feature
            feature_dict[feature] += 1
        flag += 1
        View_Bar(flag, len_lines)

    # 对字典排序
    feature_dict = sorted(feature_dict.iteritems(), key=lambda asd: asd[1], reverse=True)  # 对value进行降序排序

    print ('原始feature数目: ' + str(len(lines)))
    print ('放到dict中的数目：' + str(len(feature_dict)))

    # 将feature字典保存成文件
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/feature_dict1.csv', 'w')
    for item in feature_dict:
        f.write(str(item[0]) + ',' + str(item[1]) + '\n')

    return 0





#   只筛选 餐厅 类型的服务行业
# filter_business_id = Filter_Business('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json', "Restaurants")
#   保存review
# Filter_Business_Review('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json', filter_business_id)

#   词性标注
# Tag_Word('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/review_text.txt')
#   筛选feature词汇
# Featuer_Word('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/word_tagged_sentences.txt', window=5)

#  对feature自会进行在清洗
Feature_Data_Cleaning('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/feature.txt', stemmer=True)


