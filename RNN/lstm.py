#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/29

VECTOR_DIR = 'vectors.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.3

def evaluation(pred, label):
    count = 0
    lens = len(pred)
    for i in range(lens):
        if pred[i] == label[i]:
            count += 1
    return float(count) / lens

def precision_recall_f1(predict_result, actual_result, POSITIVE = 1):
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

if __name__ == '__main__':

    print('(1) load texts...')
    train_texts = open('train_text.txt', encoding='UTF-8').read().split('\n')
    train_labels = open('train_label.txt', encoding='UTF-8').read().split('\n')
    test_texts = open('test_text.txt', encoding='UTF-8').read().split('\n')
    test_labels = open('test_label.txt', encoding='UTF-8').read().split('\n')
    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels


    print('(2) doc to var...')
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    import numpy as np

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(all_labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)


    print('(3) split data set...')
    p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
    p2 = int(len(data)*(1-TEST_SPLIT))
    x_train = data[:p1]
    y_train = labels[:p1]
    x_val = data[p1:p2]
    y_val = labels[p1:p2]
    x_test = data[p2:]
    y_test = labels[p2:]
    print('train docs: '+str(len(x_train)))
    print('val docs: '+str(len(x_val)))
    print('test docs: '+str(len(x_test)))



    print('(5) training model...')
    from keras.layers import Dense, Input, Flatten, Dropout
    from keras.layers import LSTM, Embedding
    from keras.models import Sequential
    from keras.models import load_model

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()
    #
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['acc'])
    # print(model.metrics_names)
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    # model.save('lstm.h5')
    model = load_model('lstm.h5')
    print('(6) testing model...')
    # print(model.evaluate(x_test, y_test))
    temp = model.predict(x_test)
    pred = np.argmax(temp, axis=1)
    # print(evaluation(pred, y_test.tolist()))
    # print(*precision_recall_f1(pred.tolist(), y_test.tolist()))
    # print(type(pred))
    # print(pred)



        




