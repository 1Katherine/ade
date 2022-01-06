#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yyq_bo 
@File ：corr标准归一化.py
@Author ：Yang
@Date ：2022/1/6 18:02 
'''
import numpy as np

# 归一化
def normalization(data):
    _range = np.max(abs(data))
    return data / _range

# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 原数组
coef_ = np.array([-4.65287635e+00,  1.70422051e+02 , 2.44243495e+02 , 9.97109537e+01,
 -2.96786370e+00 ,-0.00000000e+00 , 1.09006101e+02,  2.88949014e-07,
  2.72874846e+01 ,-6.96634010e+01])
# 归一化后的数组
no_corr = np.array([-1.90501547e-02,  6.97754720e-01  ,1.00000000e+00 , 4.08244051e-01,
 -1.21512497e-02 ,-0.00000000e+00 , 4.46300941e-01 , 1.18303668e-09,
  1.11722462e-01 ,-2.85221111e-01])

print('原数组')
print(str(coef_))

print('数组标准化')
std = standardization(coef_)
print(std)

print('数组归一化')
print(normalization(coef_))

# 原数组升序排序，返回索引
print('原数组升序排列返回索引：' + str(coef_.argsort()))
# 归一化后的数组排序不变
print('归一化数组升序排列返回索引：' + str(no_corr.argsort()))
# 标准化的数组顺序不会发生变化
print('标准化数组升序排列返回索引：' + str(std.argsort()))