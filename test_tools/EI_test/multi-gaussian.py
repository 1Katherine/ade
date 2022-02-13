#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi-gaussian.py   
@Author ：Yang 
@CreateTime :   2022/1/28 16:49 
@Reference : 
'''
 
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from scipy.optimize import minimize


# # 多维高斯回归
def y_2d(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.sin(0.5 * np.linalg.norm(x, axis=1))
    y += np.random.normal(0, noise_sigma, size=y.shape)
    return y

# bounds = np.array([[4.15938070e+00,5.84061930e+00],[1.22897293e+09, 1.68142316e+09], [1.07374182e+09, 1.37179853e+09]])
bounds = np.array([[4.15938070e+00,5.84061930e+00], [3.84000000e+02, 5.43870631e+02], [4.22493703e+01, 8.57506297e+01],
                   [7.99043380e+00, 1.00000000e+01], [5.00000000e-01 ,6.26577145e-01], [4.36963171e+02, 8.95036829e+02],
                   [2.18680731e+01 ,3.61319269e+01]
                   ])

# ----------- 如果两个维度的取值差别不大，每个样本的ei值就会不同 start ------------
# 如果两个维度的取值差别不大
# train_X = np.random.uniform(-4, 4, (100, 2)).tolist()
# x_dim1 = np.linspace(-4, 4, num=100, endpoint=True).reshape(-1,1)
# np.random.shuffle(x_dim1)
# x_dim2 = np.linspace(-4, 4, num=100, endpoint=True).reshape(-1,1)
# train_X = np.hstack((x_dim1,x_dim2)).tolist()
# train_y = y_2d(train_X, noise_sigma=1e-4)
# print('train_X.len = ' + str(len(train_X)) + ' , train_X = \n' + str(train_X))
# test_d1 = np.arange(-5, 5, 0.2)
# test_d2 = np.arange(-5, 5, 0.2)
# ----------- 如果两个维度的取值差别不大，每个样本的ei值就会不同 end ------------


# ------------------- 如果两个维度的取值差别很大，每个样本的ei值就会相同 start ---------------
# for dim in range(bounds.shape[0]):
#     x_dim1 = np.linspace(bounds[dim:,0], bounds[dim:,1], num=100, endpoint=True).reshape(-1,1)
#     print(x_dim1)
# x_dim1 = np.linspace(-4, 4, num=100, endpoint=True).reshape(-1,1)
# np.random.shuffle(x_dim1)
# x_dim2 = np.linspace(80002, 80010, num=100, endpoint=True).reshape(-1,1)
# x_dim3 = np.linspace(100000, 100100, num=100, endpoint=True).reshape(-1,1)
# train_X = np.hstack((x_dim1,x_dim2,x_dim3)).tolist()
train_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(1000, bounds.shape[0]))
train_y = y_2d(train_X, noise_sigma=1e-4)
print('train_X = \n' + str(train_X))
# test_d1 = np.arange(-5, 5, 0.2)
# test_d2 = np.arange(80000, 80010, 0.2)
# ------------------- 如果两个维度的取值差别很大，每个样本的ei值就会相同 end ---------------

import random

x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(1000, bounds.shape[0]))
print('x_tries = ' + str(x_tries))
# # test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
# test_d1 = []
# test_d2 = []
# test_d3 = []
# for i in range(10000):
#     test_d1.append(random.uniform(-4, 4))
#     test_d2.append(random.uniform(80002,80010))
#     test_d3.append(random.uniform(100000, 100100))
# test_d1 = np.array(test_d1)
# test_d2 = np.array(test_d2)
# test_d3 = np.array(test_d3)
# test_X = [[d1, d2, d3] for d1, d2, d3 in zip(test_d1.ravel(), test_d2.ravel(), test_d3.ravel())]
test_X = x_tries
print('train_X = ' + str(train_X))
print('train_y = ' + str(train_y))
print('test_X = ' + str(test_X))

# Compare to sciki-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(train_X, train_y)
mu, std = gp.predict(test_X, return_std=True)
print('mean.shape() = ' + str(mu.shape) + '\n' + str(mu))
print('std.shape() = ' + str(std.shape) + '\n' + str(std))


# --------------EI_test start-------------
from scipy.stats import norm
xi = 0.1
y_max = np.max(train_y)
a = (mu - y_max - xi)
z = a / std
upper = a * norm.cdf(z) + std * norm.pdf(z)
print('upper = ' + str(upper))
print("最大值索引：", np.argmax(upper))
max_idx = upper.argmax()
print(max_idx)
x_max = test_X[max_idx]
max_acq = upper.max()
print(x_max)

# --------------EI_test end---------------

# test_y = mu.ravel()
# uncertainty = 1.96 * np.sqrt(np.diag(cov))
#
# plt.figure()
# plt.title("l=%.2f sigma_f=%.2f" % (gp.kernel_.k2.length_scale, gp.kernel_.k1.constant_value))
# plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
# plt.plot(test_X, test_y, label="predict")
# plt.scatter(train_X, train_y, label="train", c="red", marker="x")
# plt.legend()