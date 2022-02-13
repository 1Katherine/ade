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


# 不标准化
test_X_temp = test_X
train_X_temp = train_X
train_y_temp = train_y


# Compare to sciki-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(train_X_temp, train_y_temp)
mu, std = gp.predict(test_X_temp, return_std=True)
print('mu = \n' + str(mu))
predict_target = gp.predict(test_X_temp)
print('predict_target = ' + str(predict_target))
print(np.min(predict_target))
Tconstraint = np.percentile(predict_target, 25)
# Tconstraint = np.max(predict_target)
print('Tconstraint = ' + str(Tconstraint))

# --------------EI_test start-------------
from scipy.stats import norm
xi = 0.01
y_max = np.max(train_y_temp)
a = (mu - y_max - xi)
z = a / std
upper = a * norm.cdf(z) + std * norm.pdf(z)
print('upper = ' + str(upper))
# print("每一列的最大值索引：", np.argmax(upper, axis=0))
max = upper.argmax()
sortnumber = np.argsort(upper)
print('sortnumber = ' + str(sortnumber))

idx = 0
while idx < sortnumber.shape[0] - 1 and predict_target[idx] > Tconstraint:
    idx = idx + 1
    print('idx = ' + str(idx))

if idx == sortnumber.shape[0] - 1:
    x_max = test_X[upper.argmax()]
    min_pre_target = predict_target[upper.argmax()]
else :
    # print('mu[idx] = ' + str(mu[idx]))
    x_max = test_X[idx]
    min_pre_target = predict_target[idx]
print('idx = ' + str(idx) + ' , x_max = ' + str(x_max))
print('min_pre_target = ' + str(min_pre_target))