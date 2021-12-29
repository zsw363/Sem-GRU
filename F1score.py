import csv

max_len = 128


q = 0
with open('datas\\test.txt', encoding='utf-8') as f:
    c = 0
    datas = f.readlines()
    for data in datas:
        data = data.rstrip('\n')
        if data != '':
            c += 1
    print("共：%d个token"%c)
    f.close()

label_dict = {
               'O':1, 'B':2, 'I':3,
             }
real_labels = []
label = []
with open('datas\\test.txt',encoding='utf-8') as f:
    datas = f.readlines()
    for n, data in enumerate(datas):
        if len(label) >= max_len:
            if data == '\n' or n == len(datas)-1:
                assert len(label) == max_len
                real_labels.append(label)
                label = []
            continue
        if data == '\n':
            if len(label) == 0:
                continue
            while len(label) < max_len:
                label.append('[SPC]')
            assert len(label) == max_len
            real_labels.append(label)
            label = []
            continue
        data_list = data.rstrip('\n').split(' ')[1]
        label.append(str(label_dict[data_list]))
        if n == len(datas)-1:
            while len(label) < max_len:
                label.append('[SPC]')
            assert len(label) == max_len
            real_labels.append(label)
    f.close()

csv.register_dialect('mydialect',delimiter='\t',quoting=csv.QUOTE_ALL)
test_labels = []
with open('test_results.tsv',) as csvfile:
    file_list = csv.reader(csvfile,'mydialect')
    for line in file_list:
        test_labels.append(line)
csv.unregister_dialect('mydialect')

print(test_labels)
import numpy as np
print(np.array(real_labels))
print(np.array(real_labels).shape)


TP, FP, FN, TN = 0, 0, 0, 0
pos = {}
neg = {}
for i, sec in enumerate(real_labels):
    for j, token in enumerate(sec):
        # print(real_labels[i][j],test_labels[i][j])
        if token in ['[SPC]']:
            pass
        elif token == '1':
            neg[(i, j)] = token
        else:
            pos[(i, j)] = token
print(q)
print(len(neg))
print(len(pos))
for position in pos.keys():
    i, j = position
    if test_labels[i][j] == pos[position]:
        TP += 1
    else:
        FN += 1
for position in neg.keys():
    i, j = position
    if test_labels[i][j] == neg[position]:
        TN += 1
    else:
        FP += 1

precision = TP/(TP+FP)
recall = TP/(TP+FN)
print(TP, FP, FN, TN)
try:
    f1score = 2*precision*recall/(precision+recall)
except ZeroDivisionError:
    f1score = 0
print('precision:%.02f\nrecall:%.02f\nf1score:%.02f\n'%(precision,recall,f1score))