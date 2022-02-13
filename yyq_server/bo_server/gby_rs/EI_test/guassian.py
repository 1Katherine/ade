#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   guassian.py   
@Author ：Yang 
@CreateTime :   2022/1/28 14:27 
@Reference : 
'''
 
# example of a gaussian process surrogate function
import math
import numpy as np
from matplotlib import pyplot as plt
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor


# objective function
def objective(x):
    return x[0] + x[1] + x[2]


# uppper confidence bound
# beta = 1
def acquisition(X, gp):
    yhat, std = gp.predict(X, return_std=True)
    yhat = yhat.flatten()
    upper = yhat + std
    print('upper = ' + str(upper))
    max_at = np.argmax(upper)
    print('X = ' + str(X) + ' ,ucb = ' + str(X[max_at]))
    return X[max_at]

def acquisition_ei(X, ymax, gp):
    from scipy.stats import norm
    xi = 0.01
    yhat, std = gp.predict(X, return_std=True)
    print('yhat = ' + str(yhat))
    yhat = yhat.flatten()
    print('yhat flatten = ' + str(yhat))
    a = (yhat - ymax - xi)
    z = a / std
    upper = a * norm.cdf(z) + std * norm.pdf(z)
    print('upper = ' + str(upper))
    max_at = np.argmax(upper)
    print('X = ' + str(X) + ' ,ei = ' + str(X[max_at]))
    return  X[max_at]



gp = GaussianProcessRegressor()


def plot(X, y, xsamples, ysamples, yhat, std, new_x, new_y, i):
    plt.figure(figsize=(12, 6))
    plt.plot(X, y, label='real')

    plt.scatter(xsamples, ysamples, label='explored samples')
    plt.plot(X, yhat, label='gussian process - mean', c='g')
    plt.plot(X, yhat + std, label='gussian process - upper/lower bound', c='g', linestyle='--', )
    plt.plot(X, yhat - std, c='g', linestyle='--', )
    plt.scatter([new_x], [new_y], label='next sample', c='r')
    plt.legend()
    plt.title(f'Iteration {i}')
    plt.show()


if __name__ == '__main__':
    # grid-based sample of the domain [0,1]
    x1 = np.arange(0, 1, 0.01).reshape(-1,1)
    x2 = np.arange(1, 2, 0.01).reshape(-1,1)
    x3 = np.arange(2, 3, 0.01).reshape(-1,1)
    X = np.hstack((x1,x2,x3))
    print(X)



    # X = X.reshape(-1,1)
    # # sample the domain without noise
    y = np.array([objective(x) for x in X])
    y = y.reshape(-1,1)
    print(y)
    # # sample the domain with noise
    # ynoise = [objective(x) for x in X]
    # # find best result
    ix = np.argmax(y)
    xsamples=np.array([[0,1,2],[1,2,3]])
    ysamples=np.array([objective(x) for x in xsamples])

    print('xsamples ' + str(xsamples))
    print('ysamples ' + str(ysamples))
    plt.figure(figsize=(12,6))
    plt.plot(X[:,0:1].flatten(), y)
    plt.annotate('Optima',(X[ix], y[ix]))
    plt.scatter(xsamples, ysamples)
    plt.show()
    #
    # for i in range(5):
    #     gp.fit(xsamples, ysamples)
    #     yhat, std=gp.predict(X, return_std=True)
    #     std=std.reshape(-1,1)
    #     yhat=yhat.reshape(-1,1)
    #     #step
    #     ymax = np.max(ysamples)
    #     new_x=acquisition_ei(X, ymax, gp)
    #     new_y=objective(new_x)
    #
    #     new_x = new_x[:,0:1].flatten()
    #
    #     plot(X[:,0:1].flatten(), y, xsamples, ysamples, yhat, std, new_x, new_y, i)
    #     #print(f'max y is {max(ysamples.flatten())}')
    #     xsamples=np.vstack((xsamples, new_x))
    #     ysamples=np.vstack((ysamples, new_y))