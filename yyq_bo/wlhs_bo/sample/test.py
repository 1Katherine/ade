#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yyq_bo 
@File ：test.py
@Author ：Yang
@Date ：2021/12/31 15:37 
'''

# import math
# corr = 0.9
# C = 0.9
#
# r = -math.log(0.0000000001) / corr * C
# print(r)
#
# try:
#     ans = math.exp(200000)
# except OverflowError:
#     ans = float('inf')
# print(ans)
import numpy as np

a = np.array([0.83979209, 0.67087542, 0.59934671, 0.51295867, 0.79992406])
print(a.tolist())
temp = np.empty([5])
print(temp)