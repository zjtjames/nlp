#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/11

import pandas as pd

xls_file = pd.ExcelFile("remove_dup.xls")
sn = xls_file.sheet_names[0]
# 得到DataFrame
table = xls_file.parse(sn)
content_list = table['content']
content_list_new = []
for line in content_list:
    content_list_new.append(line.replace('\n', ''))
table['content'] = content_list_new
writer = pd.ExcelWriter("remove_break.xls")
table.to_excel(writer, "Sheet1")
writer.save()

