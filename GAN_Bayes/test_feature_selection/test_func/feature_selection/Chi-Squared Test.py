import csv
from pandas.core.frame import DataFrame
import pandas as pd

# ----------------------- 数据处理：csv转换成df ---------------------
tmp_lst = []
with open('generationConf.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tmp_lst.append(row)
df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
# print(df)
params_names = df.columns.tolist()[2:]
data = df[params_names]
target = df['target']
data = data[:5]
target = target[:5]
print('data = \n' + str(data))
print('target = \n' + str(target))

# ----------------------- 数据处理：标准化 ---------------------
# from sklearn.preprocessing import StandardScaler
# # # 标准化，返回值为标准化后的数据
# # data = StandardScaler().fit_transform(data)
# # data = pd.DataFrame(data, columns=params_names)
# # print(data)

# ----------------------- 数据处理：归一化 ---------------------
from sklearn.preprocessing import MinMaxScaler
#区间缩放，返回值为缩放到[0, 1]区间的数据
data = MinMaxScaler(feature_range=(0,10)).fit_transform(data)
data = pd.DataFrame(data, columns=params_names)
print(data)

# ----------------------- 特征选择：卡方检验 ---------------------
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#选择K个最好的特征，返回选择特征后的数据
after_sesctions = SelectKBest(chi2, k=2).fit_transform(data, target)
print('after_sesctions = \n' + str(after_sesctions))

# ----------------------- 特征选择输出：输出特征选择出来的特征名称 ---------------------
import operator
for col in range(after_sesctions.shape[1]):
    a = after_sesctions[: , col]
    for name in params_names:
        b = data[name].tolist()
        theList = list(set(operator.eq(b, a)))
        if True == theList[0]:
            print(name)
            break
