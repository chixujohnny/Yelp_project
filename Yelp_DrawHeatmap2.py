# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from urllib2 import urlopen
import seaborn as sns
from sklearn import preprocessing

#--------------------------------------------------#

#  This is a simple example of creating a heatmap using by Matplotlib

#--------------------------------------------------#

# column_lables = list('ABCD')
# row_labels = list('1234')
# data = np.random.rand(4, 4) # create random data
# print data
#
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
#
# # put the major ticks at the middle of each cell
# ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
# ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
#
# # want a more natural, table-like display
# ax.invert_yaxis()
# ax.xaxis.tick_top()
#
# ax.set_xticklabels(row_labels, minor=False)
# ax.set_yticklabels(column_lables, minor=False)
#
# plt.show()


#--------------------------------------------------#

#  This is another example of Basketball

#--------------------------------------------------#

# page = urlopen("http://datasets.flowingdata.com/ppg2008.csv")
# nba = pd.read_csv(page, index_col=0)
# print nba
#
# # Normalize data columns
# nba_norm = (nba - nba.mean()) / (nba.max() - nba.min())
# print nba_norm
#
# # Sort accroding to 'PTS'
# nba_sort = nba_norm.sort('PTS', ascending=True)
# print nba_sort
# print nba_sort['PTS'].head(10)
#
# # Plot is out
# fig, ax = plt.subplots() # create a picture
# heatmap = ax.pcolor(nba_sort, cmap=plt.cm.Blues, alpha=0.8)
#
# # Format
# fig = plt.gcf()
# fig.set_size_inches(8, 11)
#
# # Turn off the frame
# ax.set_frame_on(False)
#
# # put the major ticks at the middle of each cell
# ax.set_yticks(np.arange(nba_sort.shape[0]) + 0.5, minor=False)
# ax.set_xticks(np.arange(nba_sort.shape[1]) + 0.5, minor=False)
#
# # want a more natural, table-like display
# ax.invert_yaxis()
# ax.xaxis.tick_top()
#
# # Set the labels
# # label source:https://en.wikipedia.org/wiki/Basketball_statistics
# labels = [
#     'Games', 'Minutes', 'Points', 'Field goals made', 'Field goal attempts', 'Field goal percentage', 'Free throws made', 'Free throws attempts', 'Free throws percentage',
#     'Three-pointers made', 'Three-point attempt', 'Three-point percentage', 'Offensive rebounds', 'Defensive rebounds', 'Total rebounds', 'Assists', 'Steals', 'Blocks', 'Turnover', 'Personal foul']
#
# # Set label and index
# ax.set_xticklabels(labels, minor=False)
# ax.set_yticklabels(nba_sort.index, minor=False)
#
# # Rotate the labels
# plt.xticks(rotation=75)
#
# # Close the grad
# ax.grid(False)
#
# # Turn off all the ticks
# ax = plt.gca()
#
# # Turn off the scale
# for t in ax.xaxis.get_major_ticks():
#     t.tick1On = False
#     t.tick2On = False
# for t in ax.yaxis.get_major_ticks():
#     t.tick1On = False
#     t.tick2On = False
#
# # Show
# plt.show()



#--------------------------------------------------#

#  This is another example of Basketball (with seaborn)

#--------------------------------------------------#

# # Import data
# nba = pd.read_csv("http://datasets.flowingdata.com/ppg2008.csv", index_col=0)
#
# # Remove index title
# nba.index.name = ''
#
# # Normalize
# nba_norm = (nba - nba.mean()) / (nba.max() - nba.min())
#
# # Relabel columns
# labels = ['Games', 'Minutes', 'Points', 'Field goals made', 'Field goal attempts', 'Field goal percentage', 'Free throws made',
#           'Free throws attempts', 'Free throws percentage','Three-pointers made', 'Three-point attempt', 'Three-point percentage',
#           'Offensive rebounds', 'Defensive rebounds', 'Total rebounds', 'Assists', 'Steals', 'Blocks', 'Turnover', 'Personal foul']
# nba_norm.columns = labels
#
# # Set font and dpi
# sns.set(font_scale=1.2)
# sns.set_style({"savefig.dpi": 100})
#
# # Draw it
# ax = sns.heatmap(nba_norm, cmap=plt.cm.Blues, linewidths=0.1)
#
# # Set the x-axis labels on the top
# ax.xaxis.tick_top()
#
# # rotate the x-axis labels
# plt.xticks(rotation=50)
#
# # rotete the y-axis labels
# plt.yticks(rotation=0)
#
# # Get figure
# fig = ax.get_figure()
#
# # Specify dimension and show
# fig.set_size_inches(15, 20)
# plt.show()


#--------------------------------------------------#

#  Draw heatmap

#--------------------------------------------------#
def Heatmap_Data_Preprocess(data_path):

    # Import data
    data = pd.read_csv(data_path, index_col=0)
    columns = data.columns
    data = np.array(data)
    data = pd.DataFrame(data, columns=columns)

    # 数据标准化(旧)
    # data = (data - data.mean()) / (data.max() - data.min())

    # 数据标准化
    # data = pd.DataFrame(preprocessing.scale(data, axis=0, with_mean=False, with_std=False), columns=columns)

    # 数据正则化
    data = pd.DataFrame(preprocessing.normalize(data, norm='l2', axis=1), columns=columns)

    # 压缩数据,压缩至50行
    row = 50
    data_len = len(data)
    d1 = data_len / row # 将d1这么多行的数据压缩成一行

    # 重新制作 DataFrame
    data_compress = data.loc[0: d1].sum().to_frame().T
    i = 0
    while i < 50:

        if i != 49: # 如果 i 没到最后一行的话
            new_row = data.loc[i*d1: (i+1)*d1].sum().to_frame().T
            data_compress = pd.concat([data_compress, new_row], ignore_index=True)

        else: # 到了最后一行
            new_row = data.loc[i*d1: data_len-1].sum().to_frame().T
            data_compress = pd.concat([data_compress, new_row], ignore_index=True)

        i += 1

    return data_compress.iloc[:, :50] # 暂时选取前50个feature


def Heatmap_Draw(data_norm):

    # Set font and dpi
    sns.set(font_scale=1)
    sns.set_style({"savefig.dpi": 100})

    # Draw it
    ax = sns.heatmap(data_norm, cmap=plt.cm.Reds, linewidths=0.05)

    # Set the x-axis labels on the top
    ax.xaxis.tick_top()

    # rotate the x-axis labels
    plt.xticks(rotation=50)

    # rotete the y-axis labels
    plt.yticks(rotation=0)

    # Get figure
    fig = ax.get_figure()

    # Show it
    plt.show()




#
#   main
#

# Business_Feature Heatmap
# data_compress = Heatmap_Data_Preprocess('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_Business_Feature_Vector.csv')
# Heatmap_Draw(data_compress)

# User_Feature Heatmap
# data_compress = Heatmap_Data_Preprocess('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/business_Nightlife/Nightlife_User_Feature_Vector.csv')
# Heatmap_Draw(data_compress)














