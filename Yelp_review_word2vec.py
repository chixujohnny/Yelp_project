# coding:utf-8

import sys
import gensim
import nltk
import logging # 开启日志
import os.path
from gensim import corpora, models, similarities


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
#  gensim 去停用词,生成稀疏文档向量
#
def Yelp_review_preprocess(path_review): # path_review:保存顾客评论的文件路径
    lines = open(path_review, 'r').readlines()
    len_lines = len(lines)

    # 将停用词表载入字典,使用字典是因为查询速度O(1)
    stop_words_dict = {}
    stop_words = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/stop_words_eng.txt', 'r').readlines()
    for stop_word in stop_words:
        stop_word = stop_word[:-1]
        stop_words_dict[stop_word] = 1 # 把key-value扔进字典,value的值随便赋的
    print '停用词表载入字典成功!'

    # 分词、去停用词
    print '分词进度:'
    flag = 0
    reviews = []
    for line in lines:
        sentences = nltk.sent_tokenize(line) # 将文本拆分成句子列表
        for sentence in sentences:
            review_sentence = []
            words = sentence.lower().split() # ['Excellent', 'food', '.']
            for word in words: # 去停用词
                if stop_words_dict.has_key(word) == False: # 这个词不是停用词的话
                    review_sentence.append(word)
        reviews.append(review_sentence)
        flag += 1
        View_Bar(flag, len_lines)

#
#  原文-->字典
#
def Review_Saveto_Dictionary(review):
    dictionary = corpora.Dictionary(reviews)
    dictionary.save('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/review_dictionary.dict') # 把字典保存起来,方便以后使用
    print dictionary # 输出字典key的数目
    # print dictionary.token2id # 字典key-value

    return 0


#
#  生成稀疏文档向量
#
def Review_Saveto_doc2bow(review):
    # 生成稀疏文档向量
    # 函数doc2bow()对不同的单词进行了统计计数,并将单词转换成编号,以稀疏向量的形式转换成结果,例如:[[(0,1),(1,1),(2,1)], [(9,1),(10,1)]]
    dictionary = corpora.Dictionary(reviews)
    corpus = [dictionary.doc2bow(review) for review in reviews]
    corpora.MmCorpus.serialize('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/gensim_corpora.mm', corpus) # 存入硬盘,以后使用
    print '生成稀疏文档向量成功!'

    return 0


#
#  使用迭代器读取语料库
#
class Load_Corpus_with_Iteration(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line.split()




