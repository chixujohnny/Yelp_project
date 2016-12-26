# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import axes
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

tls.set_credentials_file(username='chixujohnny', api_key='O7KF0vaHoFqNnIaHE0QM')

#
#  读取文件, 输出 numpy 矩阵, feature
#
def Load_Data(Array_Path, Feature_Path):
    print '读取矩阵....'

    z = []
    x = []
    y = []
    lines = open(Array_Path, 'r').readlines()
    for item in lines:
        array = item.replace('\n', '').split(',')[1:] # 第一个成员是ID号,忽略
        ID = item.replace('\n', '').split(',')[0]
        z.append(array)
        y.append(ID)

    lines = open(Feature_Path, 'r').readlines()
    for i, item in enumerate(lines):
        x.append(item.replace('\n', ''))
        if i == 499:
            break

    print 'Done!'
    return z, x, y


#
#  绘制热图  Demo
#
def Draw_Heatmap_Demo(z, x, y):
    print '绘制热图....'

    data = [
        go.Heatmap(
            z = z,
            x = x,
            y = y,
            colorscale='Viridis',
        )
    ]

    layout = go.Layout(
        title = '用户偏好分布热图',
    )

    fig = go.Figure(data = data, layout = layout)
    py.plot(fig, filename='用户偏好分布热图')



#
#  main
#
z, x, y = Load_Data('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Business_Feature_Vector.txt', '/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Feature.txt')
Draw_Heatmap_Demo(z, x, y)

print 'ok!',
