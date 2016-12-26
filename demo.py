# coding:utf-8

import json
import chardet

#
#  json 格式的读取
#
json_input = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json', 'r').readline()
decoded = json.loads(json_input)
print json.dumps(decoded, sort_keys=True, indent=4) # 属性indent是缩进
print decoded["categories"]

json_input = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json', 'r').readline()
decoded = json.loads(json_input)
print json.dumps(decoded, sort_keys=True, indent=4) # 属性indent是缩进


#
#  字典的操作
#
d = {'a':12, 'b':3}
print d['a']
for item in d:
    print item

d = {}
item = 'a'
d[item] = 12
print d

item = "I'm afraid"
if ' ' in item:
    print 'OK!'

a = {'asd':[1,2], 'qwe':[3,4]}
for item in a:
    print item

#
#  字符串操作
#
f = open('/Users/John/Desktop/yelp_dataset_challenge_academic_dataset/demo.csv', 'w')
string = u'Books, Mags, Music & Video'
if ',' in string:
    string = string.replace(',', ' &')
    line = str(string) + ',' + str(12)
    print line

review = ['UsFtqoBl7naz8AVUBZMjQQ:This product is pretty good!']
print review[0][23:]
print review[0][:22]

text = 'Excellent food.'
text = text.decode('ascii').encode('utf-8')
print chardet.detect(text)
text = 'Excellent food.'
print chardet.detect(unicode(text, 'ascii').encode('utf-8'))
print chardet.detect(u'Excellent food.'.encode('utf-8'))

print "你好".decode('gbk').encode('utf-8')


a = "uK8tzraOp4M5u3uYrqIBXg-UsFtqoBl7naz8AVUBZMjQQ:cool"
print a[:22]
print a[23:45]
print a[46:]

a = 'This is a excellent food.'
if 'food' in a:
    print 'ok'

a = [0]*10
print a

a = 'This is a text sentence.'
if 'This is' in a:
    print 'ok'

a = [1,2,3,1,2]
print str(a)