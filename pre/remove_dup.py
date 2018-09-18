#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/11

import pandas as pd

xls_file = pd.ExcelFile("../comments.xls")
sn = xls_file.sheet_names[0]
# 得到DataFrame
table = xls_file.parse(sn)
content_list = table['content']
content_list_new = []
last_line = ''
for line in content_list:
    if line[0:int(len(line)*0.8)] != last_line[0:int(len(line)*0.8)]:
        last_line = line
        content_list_new.append(line)
    else:
        content_list_new.append('')
table['content'] = content_list_new
writer = pd.ExcelWriter("remove_dup.xls")
table.to_excel(writer, "Sheet1")
writer.save()



