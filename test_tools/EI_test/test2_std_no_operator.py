#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EI_test.py
@Author ：Yang 
@CreateTime :   2022/1/28 22:08 
@Reference : 
'''

import numpy as np

# ------------------ 设置终端输出行数 ----------------------
import pandas as pd
#显示所有列
pd.set_option('display.max_columns', 10000000)
#显示所有行
pd.set_option('display.max_rows', 1000000000)
#设置value的显示为100，默认为50
pd.set_option('max_colwidth',10000000)
pd.set_option('display.width', 1000000)

# ------------------------------------------------------------




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
test_X = []
with open("test_X.txt", "r") as f:  # 打开文件
    test_X = f.read()  # 读取文件
    test_X = eval(test_X)


# 不标准化
test_X_temp = test_X
train_X_temp = train_X
train_y_temp = train_y
f = open("./out.txt", "w")
# print('shape = ' + str(len(train_X_temp)) + ' , dim = ' + str(len(train_X_temp[0])) + ' , train_X_temp = ' + str(train_X_temp), file=f)
# print('dim = ' + str(len(train_y_temp))  +  ' , train_y_temp = ' + str(train_y_temp), file=f)
# print('shape = ' + str(len(test_X_temp)) + ' , dim = ' + str(len(test_X_temp[0])) + ' , test_X_temp = ' + str(test_X_temp), file=f)

# Compare to sciki-learn
from sklearn.gaussian_process import GaussianProcessRegressor




gp = GaussianProcessRegressor()
gp.fit(train_X_temp, train_y_temp)
mu, std = gp.predict(test_X_temp, return_std=True)
print('mu = ' + str(mu))
print('mu = \n' + str(mu), file=f)
predict_target = gp.predict(test_X_temp)
print('predict_target = ' + str(predict_target), file=f)
print(np.min(predict_target))
Tconstraint = np.percentile(predict_target, 25)
# Tconstraint = np.max(predict_target)
print('Tconstraint = ' + str(Tconstraint), file=f)



# --------------EI_test start-------------
from scipy.stats import norm
xi = 0.01
y_max = np.max(train_y_temp)
a = (mu - y_max - xi)
z = a / std
upper = a * norm.cdf(z) + std * norm.pdf(z)
print('upper = ' + str(upper), file=f)
# print("每一列的最大值索引：", np.argmax(upper, axis=0))
max = upper.argmax()
sortnumber = np.argsort(upper)
print('sortnumber = ' + str(sortnumber), file=f)

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
print('idx = ' + str(idx) + ' , x_max = ' + str(x_max), file=f)
print('min_pre_target = ' + str(min_pre_target), file=f)