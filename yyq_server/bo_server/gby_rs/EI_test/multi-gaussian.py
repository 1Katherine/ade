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


class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[
                1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

# 一维高斯回归
# def y(x, noise_sigma=0.0):
#     x = np.asarray(x)
#     y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
#     return y.tolist()
#
# train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
# train_y = y(train_X, noise_sigma=0.1)
# test_X = np.arange(0, 10, 0.1).reshape(-1, 1)
#
# gpr = GPR(optimize=True)
# gpr.fit(train_X, train_y)
# mu, cov = gpr.predict(test_X)
# test_y = mu.ravel()
# uncertainty = 1.96 * np.sqrt(np.diag(cov))
# plt.figure()
# plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
# plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
# plt.plot(test_X, test_y, label="predict")
# plt.scatter(train_X, train_y, label="train", c="red", marker="x")
# plt.legend()
# plt.show()


#
# # 多维高斯回归
def y_2d(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.sin(0.5 * np.linalg.norm(x, axis=1))
    y += np.random.normal(0, noise_sigma, size=y.shape)
    return y

# ----------- 如果两个维度的取值差别不大，每个样本的ei值就会不同 start ------------
# 如果两个维度的取值差别不大
train_X = np.random.uniform(-4, 4, (100, 2)).tolist()
train_y = y_2d(train_X, noise_sigma=1e-4)
print('train_X = \n' + str(train_X))
test_d1 = np.arange(-5, 5, 0.2)
test_d2 = np.arange(-5, 5, 0.2)
# ----------- 如果两个维度的取值差别不大，每个样本的ei值就会不同 end ------------


# ------------------- 如果两个维度的取值差别很大，每个样本的ei值就会相同 start ---------------
# x_dim1 = np.linspace(-4, 4, num=100, endpoint=True).reshape(-1,1)
# x_dim2 = np.linspace(80002, 80010, num=100, endpoint=True).reshape(-1,1)
# train_X = np.hstack((x_dim1,x_dim2)).tolist()
# train_y = y_2d(train_X, noise_sigma=1e-4)
# print('train_X = \n' + str(train_X))
# test_d1 = np.arange(-5, 5, 0.2)
# test_d2 = np.arange(80000, 80010, 0.2)
# ------------------- 如果两个维度的取值差别很大，每个样本的ei值就会相同 end ---------------


test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]
#
# gpr = GPR(optimize=False)
# gpr.fit(train_X, train_y)
# mu, cov = gpr.predict(test_X)
# z = mu.reshape(test_d1.shape)
#
# fig = plt.figure(figsize=(7, 5))
# ax = Axes3D(fig)
# ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
# ax.scatter(np.asarray(train_X)[:, 0], np.asarray(train_X)[:, 1], train_y, c=train_y, cmap=cm.coolwarm)
# ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
# ax.set_title("with optimization l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
# plt.show()


# Compare to sciki-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(train_X, train_y)
mu, std = gp.predict(test_X, return_std=True)
print('predict = \n' + str(mu))

# --------------EI_test start-------------
from scipy.stats import norm
xi = 0.01
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