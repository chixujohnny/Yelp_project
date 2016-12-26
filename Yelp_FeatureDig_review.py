# coding: utf-8

#####################################
#  对标注好词性的review进行feature抽取  #
#####################################

import os, sys

#  进度条
def View_Bar(flag, sum):
    rate = float(flag) / sum
    rate_num = rate * 100
    if flag % 15.0 == 0:
        print '\r%.2f%%: ' % (rate_num),  # \r%.2f后面跟的两个百分号会输出一个'%'
        sys.stdout.flush()

def Filter_Feature(tagged_reviews, window):

    tagged_sentences = [] # 标注好的所有句子
    feature_list = [] # 挖到的 feature
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

    # 数据预处理,把txt放入内存
    print '数据预处理....'
    flag = 0
    total = len(tagged_reviews)
    for line in tagged_reviews:
        line = line[23:-1].split('&&')[:-1]
        for sentence in line:
            sentence = sentence.split(' ')
            tagged_sentence = [] # 标注好的每一个句子 [('Excellent','JJ'), ('food','NN'), ('.','.')]
            for item in sentence[:-1]: # 去掉最后一个多余的成员
                item = tuple(item.split('//'))
                if item[1] != '.' and item[1] != ':' and item[1] != ',' and item[1] != '(' and item[1] != ')' and item[1] != "''" and item[1] != '``' and item[1] != '$':
                    tagged_sentence.append(item)
            if tagged_sentence != []:
                tagged_sentences.append(tagged_sentence)
        flag += 1
        View_Bar(flag, total)
    print 'Done!'

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
        item = item.lower() # 将大写字母变小写
        if feature_dict.has_key(item) == False: # 如果字典中没有这个key
            feature_dict[item] = 1
        else:
            feature_dict[item] += 1
    # 对字典排序,排序完是list嵌套tuple
    feature_dict = sorted(feature_dict.iteritems(), key=lambda asd:asd[1], reverse=True) # 降序排序
    print feature_dict

    # save
    f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Feature.txt', 'w')
    for item in feature_dict:
        f.write(item[0] + ',' + str(item[1]) + '\n')
    print('feature词汇保存完毕')





if __name__ == '__main__':

    tagged_reviews = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Review_Tagged.txt', 'r').readlines()
    Filter_Feature(tagged_reviews, window=5)