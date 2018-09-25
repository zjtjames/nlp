#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/19

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec
import gensim
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier



class Hotel(object):

    def __init__(self, data_dir, save_dir, max_iter=5000, model_type="mlp"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.max_iter = max_iter
        self.word2id = {}
        self.id2word = {}
        self.model_type = model_type

    # 序列化
    def save(self, model, name):
        with open(self.save_dir + '%s.pkl' % name, 'wb') as fid:
            pickle.dump(model, fid)

    # 反序列化
    def load(self, name):
        with open(self.save_dir + '%s.pkl' % name, 'rb') as fid:
            model = pickle.load(fid)
        self.model_type = name
        return model

    def load_txt(self, file_name):
        text = []
        with open(self.data_dir + file_name, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                text.append([x for x in line.strip().split(' ')])
        return text

    def load_label(self, file_name):
        label = []
        with open(self.data_dir + file_name, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                label.append(int(line.strip()))
        return np.asarray(label, dtype=np.float32)

    def get_text_word2vec(self, text_list, w2v_model):
        text_vec = []
        get_count = 0
        notget_count = 0
        for line in text_list:
            vec = np.zeros(100, dtype=np.float16)
            word_num = 0
            for word in line:
                if word in w2v_model:
                    vec += w2v_model[word]
                    word_num += 1
                    get_count += 1
                else:
                    notget_count += 1
            if word_num > 0:
                vec = vec / word_num
            text_vec.append(vec)
        print("get %d" % get_count)
        print("not get %d" % notget_count)
        return np.asarray(text_vec, dtype=np.float32)

    def train(self, x, y, save_name, model_type='svm'):
        if model_type == 'logistic':
            the_model = LogisticRegression(max_iter=self.max_iter)
        elif model_type == 'svm':
            the_model = LinearSVC(max_iter=self.max_iter)
        elif model_type == 'mlp':
            the_model = MLPClassifier(max_iter=self.max_iter)
        the_model.fit(x, y)
        self.save(the_model, save_name)

    def pred(self, model, x):
        return model.predict(x)

    def evaluation(self, pred, label):
        count = 0
        lens = len(pred)
        for i in range(lens):
            if pred[i] == label[i]:
                count += 1
        return float(count) / lens

    def get_key_words(self, coefs):
        coef = coefs.reshape(-1)
        topk = np.argsort(coef)[::-1]
        with open(self.save_dir + self.model_type + '_key_words.txt', 'wb') as f:
            for i in topk:
                f.write(self.id2word[i] + '	' + str(coef[i]) + '\n')


if __name__ == '__main__':
    hotel = Hotel("data/", "storage/")
    # 自己训练的w2v
    word2vec_model = word2vec.Word2Vec.load("data/hotel.model")
    # 维基百科w2v
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("data/wiki.zh.vector.bin", binary=True)

    # 训练
    train_text = hotel.load_txt("train_text.txt")
    train_text_vec = hotel.get_text_word2vec(train_text, word2vec_model)
    train_label = hotel.load_label('train_label.txt')
    # hotel.train(train_text_vec, train_label, 'SVM', model_type='svm')
    # hotel.train(train_text_vec, train_label, 'logistic', model_type='logistic')
    hotel.train(train_text_vec, train_label, 'MLP', model_type='mlp')

    # 测试
    test_text = hotel.load_txt('test_text.txt')
    test_vec = hotel.get_text_word2vec(test_text, word2vec_model)
    test_label = hotel.load_label('test_label.txt')
    # 输入加载模型的名字
    class_model = hotel.load('MLP')
    pred = hotel.pred(class_model, test_vec)  # 判断是否异常
    accuracy = hotel.evaluation(pred, test_label)
    print('test accuracy: %.3f' % accuracy)













