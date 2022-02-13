#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GaussianProcessRegressor.py   
@Author ：Yang 
@CreateTime :   2022/2/9 11:52 
@Reference : https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
'''
 
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.gaussian_process import GaussianProcessRegressor
a=np.random.random(50).reshape(50,1)
b=2000000+np.random.random(50).reshape(50,1)
plt.scatter(a,b,marker = 'o', color = 'r', label='3', s = 15)
plt.show()
gaussian=GaussianProcessRegressor()
fiting=gaussian.fit(a,b)
c=np.linspace(0.1,1,100)
mean,std=gaussian.predict(c.reshape(100,1),return_std=True)
print('std.shape() = ' + str(std.shape) + '\n' + str(std))
print('mean.shape() = ' + str(mean.shape) + '\n' + str(mean))
plt.scatter(a,b,marker = 'o', color = 'r', label='3', s = 15)
plt.plot(c,mean)
plt.show()
