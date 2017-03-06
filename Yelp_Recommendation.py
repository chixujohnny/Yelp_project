# coding: utf-8

###############################
#  基于服务价值特征分布的服务推荐  #
###############################


import numpy as np


def Cosine_Similarity(X, Y):

    X = np.array(X) # 将list转换成行向量
    Y = np.array(Y)

    num = float(X * Y.T)
    denom = np.linalg.norm(X) * np.linalg.norm(Y)

    cos = num / denom # 余弦值

    sim = 0.5 + 0.5 * cos # 归一化

    return sim


def Data_Preprocess(Business_Vector_Path, User_Vector_Path):

    Business_Vector_Dict = {}
    lines = open(Business_Vector_Path, 'r').readlines()

    for line in lines[1:]:

        data = line.split(',')
        Business_ID = data[0]
        Distribution = data[1:] # 特征分布
        Business_Vector_Dict[Business_ID] = Distribution

    # ---------------------------------- #

    User_Vector_Dict = {}
    lines = open(User_Vector_Path, 'r').readlines()

    for line in lines[1:]:

        data = line.split(',')
        User_ID = data[0]
        Distribution = data[1:]
        User_Vector_Dict[User_ID] = Distribution

    return Business_Vector_Dict, User_Vector_Dict


def Recommendation(Business_Vector_Dict, User_Vector_Dict):

    User_ID = raw_input('Please input User_ID to Recommend: ')
    User_vsm = User_Vector_Dict[User_ID]

    dist = {}
    for Business_ID in Business_Vector_Dict:

        Business_vsm = Business_Vector_Dict[Business_ID]
        dist[Business_ID] = Cosine_Similarity(Business_vsm, User_vsm)

    dist = sorted(dist.iteritems(), key=lambda asd:asd[1], reverse=True) # 字典降序排序
    print 'The top-5 recommended Business_ID is :',
    print dist[:5]