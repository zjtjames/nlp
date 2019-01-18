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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from scipy import interp
import matplotlib.pyplot as plt



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
            the_model = SVC(probability=True)
        elif model_type == 'mlp':
            the_model = MLPClassifier(max_iter=self.max_iter)
        elif model_type == 'rf':
            the_model = RandomForestClassifier(max_depth=self.max_iter)
        the_model.fit(x, y)
        self.save(the_model, save_name)

    def pred(self, model, x):
        return model.predict_proba(x)

    def evaluation(self, pred, label):
        count = 0
        lens = len(pred)
        for i in range(lens):
            if pred[i] == label[i]:
                count += 1
        return float(count) / lens

    def precision_recall_f1(self, predict_result, actual_result, POSITIVE = 1):
        if len(predict_result) != len(actual_result):
            raise ValueError("预测结果集与真实结果集大小不一致")

        '''
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = 2*TP(2*TP+FP+FN)
        '''
        if not POSITIVE:
            POSITIVE = 1 if actual_result.count(1)/len(actual_result) > 0.5 else 0
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(predict_result)):
            if predict_result[i] == POSITIVE:
                if actual_result[i] == POSITIVE:
                    TP += 1
                else:
                    FP += 1
            elif actual_result[i] == POSITIVE:
                FN += 1
            else: TN += 1

        if TP+FP == 0:
            raise ValueError("预测正类数为0")

        if TP + FN == 0:
            raise ValueError("真实正类数为0")

        return TP/(TP + FP), TP / (TP + FN), 2*TP/(2*TP+FP+FN)
        # return TP, FN, FP, TN

    def roc(self, test_y, train_y, predict):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        y_target = np.r_[train_y,test_y]
        cv = StratifiedKFold(y_target, n_folds=6)

        #画ROC曲线和计算AUC
        fpr, tpr, thresholds = roc_curve(test_y, predict[:,1], pos_label = 1)##指定正例标签，pos_label = ###########

        mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0] = 0.0                               #初始处为0
        roc_auc = auc(fpr, tpr)
        #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        #画对角线
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.plot(fpr, tpr, lw=1, label='ROC   (area = %0.3f)' %  roc_auc)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.model_type)
        plt.legend(loc="lower right")
        plt.show()


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
    # hotel.train(train_text_vec, train_label, 'MLP', model_type='mlp')
    # hotel.train(train_text_vec, train_label, 'RF', model_type='rf')

    # 测试
    test_text = hotel.load_txt('test_text.txt')
    test_vec = hotel.get_text_word2vec(test_text, word2vec_model)
    test_label = hotel.load_label('test_label.txt')
    # 输入加载模型的名字
    class_model = hotel.load('svm')
    pred = hotel.pred(class_model, test_vec)  # 判断是否异常
    # accuracy = hotel.evaluation(pred, test_label)

    # print('test accuracy: %.3f' % accuracy)
    # print(*hotel.precision_recall_f1(pred, test_label))
    hotel.roc(test_label,train_label, pred)













