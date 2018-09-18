#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Zheng Jiateng on 2018/9/12

import pandas as pd

xls_file = pd.ExcelFile("remove_break.xls")
sn = xls_file.sheet_names[0]
# 得到DataFrame
table = xls_file.parse(sn)
content_list = table['content']
content_list_new = []
for line in content_list:
    line_new = ''
    for char in line:
        if 19968 <= ord(char) <= 40869 or 48 <= ord(char) <= 57 or 97 <= ord(char) <= 122 or 65 <= ord(char) <= 90:
            line_new += char
    content_list_new.append(line_new)
table['content'] = content_list_new
writer = pd.ExcelWriter("remove_special.xls")
table.to_excel(writer, "Sheet1")
writer.save()
