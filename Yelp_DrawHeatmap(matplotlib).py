# conding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from urllib2 import urlopen
import seaborn as sns

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

page = urlopen("http://datasets.flowingdata.com/ppg2008.csv")
nba = pd.read_csv(page, index_col=0)
print nba

# Normalize data columns
nba_norm = (nba - nba.mean()) / (nba.max() - nba.min())
print nba_norm

# Sort accroding to 'PTS'
nba_sort = nba_norm.sort('PTS', ascending=True)
print nba_sort
print nba_sort['PTS'].head(10)

# Plot is out
fig, ax = plt.subplots() # create a picture
heatmap = ax.pcolor(nba_sort, cmap=plt.cm.Blues, alpha=0.8)

# Format
fig = plt.gcf()
fig.set_size_inches(8, 11)

# Turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(nba_sort.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(nba_sort.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

# Set the labels
# label source:https://en.wikipedia.org/wiki/Basketball_statistics
labels = [
    'Games', 'Minutes', 'Points', 'Field goals made', 'Field goal attempts', 'Field goal percentage', 'Free throws made', 'Free throws attempts', 'Free throws percentage',
    'Three-pointers made', 'Three-point attempt', 'Three-point percentage', 'Offensive rebounds', 'Defensive rebounds', 'Total rebounds', 'Assists', 'Steals', 'Blocks', 'Turnover', 'Personal foul']

# Set label and index
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(nba_sort.index, minor=False)

# Rotate the labels
plt.xticks(rotation=75)

# Close the grad
ax.grid(False)

# Turn off all the ticks
ax = plt.gca()

# Turn off the scale
for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

# Show
plt.show()







