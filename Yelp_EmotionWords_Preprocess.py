# coding: utf-8

############################
#  《知网》情感分析词--预处理  #
############################

import time

#  正面、反面词语预处理 --> 文件
def Preprocess_Words(path, save_path):

    f = open(save_path, 'w')

    lines = open(path, 'r').readlines()[2:] # 前两行没用
    for line in lines:
        line = line.replace('\n', '').replace('\r', '').strip() # 去掉后面没用的
        # 只保留单词,词组去掉
        if ' ' not in line:
            f.write(line + '\n')

    return 0


#  程度词语预处理 --> 内存
def Preprocess_Degree_Words(path):

    lines = open(path, 'r').readlines()
    #                    most    very    more    bit    just    over
    All_Degree_Words = [[    ], [    ], [    ], [   ], [    ], [    ]]
    cate = -1
    for line in lines:
        line = line.replace('\n', '')
        if line[0] == '=':
            cate += 1
            continue
        All_Degree_Words[cate].append(line)

    return 0




# all_path = ['/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/负面评价词语（英文）.txt',
#         '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/负面情感词语（英文）.txt',
#         '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/正面评价词语（英文）.txt',
#         '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/正面情感词语（英文）.txt']
# save_path_part = '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/知网情感分析用词语集/English/save'
#
# for i, path in enumerate(all_path):
#     save_path = save_path_part + '_' + str(i) + '.txt'
#     Preprocess_Words(path, save_path)


