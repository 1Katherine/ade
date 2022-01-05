
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：gby 
@File ：test.py
@Author ：Yang
@Date ：2022/1/4 17:41 
'''

import numpy as np
lb = np.array([5.00000000e-01,1.00000000e+00,3.00000000e+00,4.00000000e+00,
               5.00000000e-01,1.60000000e+01,1.07374182e+09,2.00000000e+02,
               2.40000000e+01,3.84000000e+02])
ub = np.array([9.00000000e-01,3.00000000e+00,7.00000000e+00,8.00000000e+00,
               9.00000000e-01,4.80000000e+01,2.14748365e+09,5.00000000e+02,
               7.20000000e+01,8.77000000e+02])
ub_extend = [9.00000000e-01 ,4.00000000e+00, 1.00000000e+01, 1.10000000e+01,
 9.00000000e-01, 7.90000000e+01 ,3.22122547e+09, 7.11000000e+02,
 8.70000000e+01 ,8.95000000e+02]
X = np.array([[0.71428571,0.         ,0.71428571, 0.71428571,0.55555556 ,0.36507937,
  0.38802752 ,0.59295499, 0.95238095, 0.69863014],
 [0.36507937, 0.33333333 ,0.57142857 ,1.,         0.23809524, 0.92063492,
  0.86781143 ,0.74363992 ,0.65079365 ,0.60273973],
 [0.07936508 ,0.66666667, 0.14285714, 0.14285714, 0.12698413 ,0.49206349,
  0.42961303 ,0.90606654, 0.28571429 ,0.12915851],
[0.9047619 , 0.        , 0.85714286, 1.   ,      0.57142857, 0.68253968,
0.98349953 ,0.4109589 , 0.65079365 ,0.16242661]])
# X = lb + (ub - lb) * X
# print(X)

print('防止越界')
X = lb + (ub_extend - lb) * X
# 如果值超过上界，则这个值等于上界，否则不变（维护参数边界）
X = np.where(X > ub, ub, X)
print(X)


# x=np.array([1,4,3,-1,6,9])
# # [3 0 2 1 4 5]
# # print(np.argsort(x))
# # [1 3 2 0 4 5]
# print(np.argsort(np.argsort(x)))