#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EI_test.py
@Author ：Yang 
@CreateTime :   2022/1/28 22:08 
@Reference : 
'''

import numpy as np
train_X = []
with open("train_X.txt", "r") as f:  # 打开文件
    train_X = f.read()  # 读取文件
    train_X = eval(train_X)
train_y = []
with open("train_y.txt", "r") as f:  # 打开文件
    train_y = f.read()  # 读取文件
train_y = train_y.split()
y_len = len(train_y)
train_y[0] = train_y[0].split('[')[1]
train_y[y_len - 1] = train_y[y_len - 1].split(']')[0]

train_y = np.array(train_y,dtype=np.float64)
train_y = [-x for x in train_y]
print(train_y)
test_X = []
with open("test_X.txt", "r") as f:  # 打开文件
    test_X = f.read()  # 读取文件
    test_X = eval(test_X)


# 归一化和标准化
# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
test_X_temp = standardization(np.array(test_X))
train_X_temp = standardization(np.array(train_X))
train_y_temp = standardization(np.array(train_y))

# 反标准化
def revurse_standardization(data, originalData):
    mu = np.mean(originalData, axis=0)
    sigma = np.std(originalData, axis=0)
    return data * sigma + mu

# 归一化
# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range
# test_X_temp = normalization(np.array(test_X))
# train_X = normalization(np.array(train_X))
# train_y = normalization(np.array(train_y))


# Compare to sciki-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(train_X_temp, train_y_temp)
mu, cov = gp.predict(test_X_temp, return_cov=True)
print('mu = \n' + str(mu))
print('reverse mu = \n' + str(revurse_standardization(mu, train_y)))
predict_target = revurse_standardization(mu, train_y)
print('Tconstraint = ' + str(np.percentile(predict_target, 25)))

# --------------EI_test start-------------
from scipy.stats import norm
xi = 0.01
y_max = np.max(train_y_temp)
a = (mu - y_max - xi)
z = a / cov
upper = a * norm.cdf(z) + cov * norm.pdf(z)
print('upper = ' + str(upper))
# print("每一列的最大值索引：", np.argmax(upper, axis=0))
max = upper.argmax()
idx = max % 1000
x_max = test_X[idx]
max_acq = upper.max()
min_pre_target = revurse_standardization(gp.predict([x_max]), train_y)
print('x_max = ' + str(x_max))
print('min_pre_target = ' + str(min_pre_target))