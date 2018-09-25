#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/24

from random import shuffle

with open('segmented.txt', 'r', encoding='UTF-8') as f:
    x = f.readlines()
with open('label.txt', 'r', encoding='UTF-8') as f:
    y = f.readlines()
# 列表生成式会产生内存error
# data = [m.strip() + n.strip() for m in x for n in y]
y_it = iter(y)
data = []
for line in x:
    label = next(y_it)
    if line.strip():
        line_new = line.strip() + ' ' + label
        data.append(line_new)
shuffle(data)
data_train = data[:int(0.7*len(data))]
data_test = data[int(0.7*len(data)):]
with open('train_text.txt', 'w', encoding='UTF-8') as f:
    for line in data_train:
        f.write(line)
with open('test_text.txt', 'w', encoding='UTF-8') as f:
    for line in data_test:
        f.write(line)




