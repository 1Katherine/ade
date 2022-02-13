#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tools.py   
@Author ：Yang 
@CreateTime :   2022/1/29 12:38 
@Reference : 
'''
import numpy as np

# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
 
# 反标准化
def reverse_standardization(data, originalData):
    mu = np.mean(originalData, axis=0)
    sigma = np.std(originalData, axis=0)
    return data * sigma + mu

# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# 反归一化
def reverse_normalization(data, originalData):
    _range = np.max(originalData) - np.min(originalData)
    return data * _range + np.min(originalData)