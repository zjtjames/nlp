#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/10/11
# 调用百度api

import requests
import time


def get_token():
    headers = {'Content-Type': 'application/json'}
    response = requests.post('https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials'
                             '&client_id=KBhjCyoiKi9wXyg0rQg4dooW'
                             '&client_secret=qR6GncV9kMbX1nWGk0fAc4CD2hwFRHVP',headers = headers)
    json = response.json()
    return json['access_token']


def query(text):
    try:
        access_token = '24.f55542ae4f8c51898e603aa4144441ec.2592000.1541820414.282335-14393081'
        headers = {'Content-Type': 'application/json'}
        body = {'text': text}
        resonse = requests.post('https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify'
                                '?charset=UTF-8&access_token='+access_token, json=body, headers = headers)
        result = resonse.json()
        return result['items'][0]['sentiment']
    except Exception as e:
        pass


if __name__ == '__main__':
    with open('F:\code\python\私活\文本分类私活\pre\segmented.txt', 'r', encoding='UTF-8') as f:
        text = f.readlines()
        # 2
        positive_count = 0
        # 0
        negative_count = 0
        # 1
        neutral_count = 0
        all_count = 0
    with open('result.txt', 'w', encoding='UTF-8') as f:
        for line in text:
            result = query(line)
            f.write(str(result) + '\n')
            print(result)
            if result == 2:
                positive_count += 1
                all_count += 1
            elif result == 0:
                negative_count += 1
                all_count += 1
            elif result == 1:
                neutral_count += 1
                all_count += 1
            time.sleep(0.15)
    print("积极情绪: %s %s" % (positive_count, positive_count/all_count))
    print("中性情绪: %s %s" % (neutral_count, neutral_count/all_count))
    print("消极情绪: %s %s" % (negative_count, negative_count/all_count))





