from scipy.optimize import minimize, rosen, rosen_der

'''
    考虑最小化罗森布拉克函数rosen function的问题, 使用Nelder-Mead方法计算最小值
'''
# 初始样本
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
# rosen函数的最小值对应的样本x
print(res.x)

# 使用BFGS方法计算最小值
res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'gtol': 1e-6, 'disp': True})
# rosen函数的最小值对应的样本x
print(res.x)