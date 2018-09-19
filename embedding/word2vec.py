#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/19

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec
import numpy as np
import pandas as pd



if __name__ == '__main__':
    sentences = word2vec.Text8Corpus("F:\code\python\私活\文本分类私活\pre\segmented.txt")
    model = word2vec.Word2Vec.load("hotel.model")
    with open(u"F:\code\python\私活\文本分类私活\pre\segmented.txt", "r", encoding='UTF-8') as f:
        vecs = []
        for line in f.readlines():
            vec = np.zeros(100).reshape((1, 100))
            count = 0
            words = line.strip().split(' ')
            for word in words[1:]:
                try:
                    count += 1
                    vec += model[word].reshape((1, 100))
                except KeyError:
                    continue
            vec /= count
            vecs.append(vec)
            break
        print(vecs)











