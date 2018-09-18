#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/8/28

import jieba
import pandas as pd


def stopwordslist(filepath):
    with open(filepath, 'r') as f:
        # 列表生成式
        stopwords = [line.strip() for line in f.readlines()]
        return stopwords


def seg_sentence(sentence, stopwords):
    sentence_seged = jieba.cut(sentence.strip(), cut_all=False)
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    return outstr

if __name__ == '__main__':
    # 得到停用词的list
    stopword = stopwordslist("stopwords.txt")
    xls_file = pd.ExcelFile("preprocessed.xls")
    sn = xls_file.sheet_names[0]
    # 得到DataFrame
    table = xls_file.parse(sn)
    content_list = table['content']
    with open('segmented.txt', 'w') as f:
        for line in content_list:
            line_seg = seg_sentence(line, stopword)
            f.write(line_seg + '\n')







