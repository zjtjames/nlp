#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/10/15

from gensim import corpora, models


with open('../pre/segmented0.txt', 'r', encoding='utf-8') as f:
    data_list = []
    # 构造词典向量
    for line in f.readlines():
        data_list.append(line.strip().split(' '))
    dict = corpora.Dictionary(data_list)
    corpus = [dict.doc2bow(text) for text in data_list]
    tfidf = models.TfidfModel(corpus) #统计tfidf
    corpus_tfidf = tfidf[corpus]  #得到每个文本的tfidf向量，稀疏矩阵
    ldamodel = models.LdaModel(corpus_tfidf, id2word=dict, num_topics=6)
    print(ldamodel.print_topics(num_topics=6, num_words=10))






