import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.gaussian_process import GaussianProcessRegressor
x=np.random.random(50).reshape(50,1)
y=x*2+np.random.random(50).reshape(50,1)
# 画出数据集散点图
plt.scatter(x,y,marker = 'o', color = 'r', label='3', s = 15)
plt.show()
# 高斯回归过程模型
gaussian=GaussianProcessRegressor()
# 拟合训练集
fiting=gaussian.fit(x,y)
# 预测c （共有100个样本点）
c=np.linspace(0.1,1,100)
print('c = \n' + str(c))
# 根据GP模型预测样本点c对应的y值（得到100个y值）
d=gaussian.predict(c.reshape(100,1))
print('d = \n' + str(d))
# 画出数据集散点图
plt.scatter(x,y,marker = 'o', color = 'r', label='3', s = 15)
# 画出曲线（高斯回归模型模型根据c，预测的y值 - d）
plt.plot(c,d)
plt.show()