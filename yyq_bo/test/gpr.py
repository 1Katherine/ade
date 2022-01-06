from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt

# sklearn用于回归任务的数据集，包含了特征的乘积和互换操作
X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
# kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(random_state=0)
# 获取此估算工具的参数
print(gpr.get_params())
# 拟合高斯过程回归模型
gpr.fit(X, y)
# 使用高斯过程回归模型进行预测
# print(X[:2,:]) 对两个样本进行预测，返回两个样本的[均值，标准差]
print('\n均值和标准差 \n' + str(gpr.predict(X[:2,:], return_std=True)))
# print(X[:2,:]) 对两个样本进行预测，返回两个样本的[均值，协方差]
print('\n均值和协方差 \n' + str(gpr.predict(X[:2,:], return_cov=True)))
# print(X[:2,:]) 对两个样本进行预测，只返回两个样本的均值
pre_y = gpr.predict(X[:2,:])
# 返回预测的确定系数R ^ 2
print('\nscore \n' + str(gpr.score(X, y)))
