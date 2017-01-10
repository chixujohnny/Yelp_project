# coding:utf-8

#############################
#  对 review 内容进行词性标注  #
#############################

import nltk, sys, os
from nltk import SnowballStemmer
import chardet

#  进度条
def View_Bar(flag, sum):
    rate = float(flag) / sum
    rate_num = rate * 100
    if flag % 15.0 == 0:
        print '\r%.2f%%: ' % (rate_num),  # \r%.2f后面跟的两个百分号会输出一个'%'
        sys.stdout.flush()


#  nltk词性标注
def Word_Tagged(review):
    print 'nltk词性标注....'
    tagged_reviews = []
    total = len(review)
    # f = open('/Users/John/Desktop/' + categories + '_Review_Tagged.txt', 'w')

    #  先创建一个哈希表保存停用词
    # stop_words_dict = {}
    # for item in stop_words:
    #     stop_words_dict[item] = 1

    # 词性标注
    flag = 0 # 进度条
    for text in review: # review = ['UsFtqoBl7naz8AVUBZMjQQ:This product is pretty good!']
        User_ID = text[:22]
        text = text[23:]
        # codec = chardet.detect(text)
        # if codec['encoding'] != 'ascii':
        #     flag += 1
        #     View_Bar(flag, total)
        #     continue
        sentences = nltk.sent_tokenize(text) # 将长文本拆成句子列表,后期词性标注精确
        # f.write(User_ID + ':')
        for sentence in sentences:
            word = nltk.word_tokenize(sentence) # 对句子进行分词
            word_tagged = nltk.pos_tag(word) # 词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]
            tagged_reviews.append(word_tagged)
            # 写文件
        #     for item in word_tagged:
        #         f.write(item[0] + '//' + item[1] + ' ') # 'Excellent//JJ food//NN .//. '
        #     f.write('&&') # 每句话的分隔符
        # f.write('\n')
        flag += 1
        View_Bar(flag, total)

    return tagged_reviews



if __name__ == '__main__':

    review = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Review.txt', 'r').readlines()
    Word_Tagged(review)


