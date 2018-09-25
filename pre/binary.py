#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/8/28

import pandas as pd

xls_file = pd.ExcelFile("preprocessed.xls")
sn = xls_file.sheet_names[0]
# 得到DataFrame
table = xls_file.parse(sn)
score_list = table['score']
bi_score_list = []
for score in score_list:
    if score > 3.0:
        bi_score_list.append(1)
    else:
        bi_score_list.append(0)
table['bi_score'] = bi_score_list
writer = pd.ExcelWriter("binary.xls")
table.to_excel(writer, "Sheet1")
writer.save()

