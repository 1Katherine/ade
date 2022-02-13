#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cdf_test.py   
@Author ：Yang 
@CreateTime :   2022/1/25 20:00 
@Reference : 
'''
from scipy.stats import norm
import random
import numpy as np
import matplotlib.pyplot as plt
x = np.array(np.random.randint(10,size=10))

cdf = norm.cdf(x)
print(x)
print(cdf)
print(np.sort(x))
print(list(zip(np.sort(x), np.sort(cdf))))
# 画图
plt.plot(x, cdf, label='cdf')
plt.scatter(x, cdf, s=20, color='r')
plt.xlabel('n_iter')
plt.ylabel('runtime')
plt.legend()
plt.show()
