#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/25
# 生成tag

with open('train_text.txt', 'r', encoding='UTF-8') as f:
    data = f.readlines()
    tag_list = []
    for line in data:
        tag = line.split(' ')[-1]
        tag_list.append(tag)
with open('train_label.txt', 'w', encoding='UTF-8') as f:
    for line in tag_list:
        f.write(line)

with open('test_text.txt', 'r', encoding='UTF-8') as f:
    data = f.readlines()
    tag_list = []
    for line in data:
        tag = line.split(' ')[-1]
        tag_list.append(tag)
with open('test_label.txt', 'w', encoding='UTF-8') as f:
    for line in tag_list:
        f.write(line)

