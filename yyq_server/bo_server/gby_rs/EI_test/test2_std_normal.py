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


# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# 反归一化
def reverse_normalization(data, originalData):
    _range = np.max(originalData) - np.min(originalData)
    return data * _range + np.min(originalData)


test_X_temp = normalization(np.array(test_X))
train_X_temp = normalization(np.array(train_X))
train_y_temp = normalization(np.array(train_y))

x_dim1 = np.array(train_X)[:,0]
print(normalization(x_dim1))

print('归一化前的train_X = \n' + str(np.array(train_X)))
print('归一化后的train_X = \n' + str(train_X_temp))

# Compare to sciki-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(train_X_temp, train_y_temp)
mu, std = gp.predict(test_X_temp, return_std=True)
print('mu = \n' + str(mu))
print('min mu = \n' + str(np.min(mu)))
Tconstraint_normal = np.percentile(train_y_temp, 75)
print('标准化后的 mu Tconstraint = ' + str(Tconstraint_normal))

predict_target = reverse_normalization(mu, train_y)
print('predict_target = ' + str(predict_target))
print(np.min(predict_target))
print('min predict_target = ' + str(np.min(predict_target)))
Tconstraint = np.percentile(train_y, 75)
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
# 降序排列
sortnumber = np.argsort(-upper)
print('sortnumber = ' + str(sortnumber))
print('upper after sortnumber = ' + str(upper[sortnumber]))
print('predict_target after sortnumber = ' + str(predict_target[sortnumber]))
print('mu after sortnumber = ' + str(mu[sortnumber]))

idx = 0
while idx < sortnumber.shape[0] - 1 and mu[sortnumber[idx]] > Tconstraint_normal:
# while idx < sortnumber.shape[0] - 1 and predict_target[sortnumber[idx]] > Tconstraint:
    print('idx = ' + str(idx) + ', upper sort idx = ' + str(sortnumber[idx]))
    idx = idx + 1

if idx == sortnumber.shape[0] - 1:
    x_max = test_X[upper.argmax()]
    min_pre_target = predict_target[upper.argmax()]
else :
    # print('mu[idx] = ' + str(mu[idx]))
    x_max = test_X[sortnumber[idx]]
    min_pre_target = predict_target[sortnumber[idx]]
print('idx = ' + str(sortnumber[idx]) + ' , x_max = ' + str(x_max))
print('min_pre_target = ' + str(min_pre_target))