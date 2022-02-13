#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   corraltion.py   
@Author ：Yang 
@CreateTime :   2022/2/13 12:36 
@Reference : 
'''
import pandas as pd
import csv
import numpy as np
params_names = []
def csvTodf():
    global params_names
    # ----------------------- 数据处理：csv转换成df ---------------------
    tmp_lst = []
    with open('generationConf.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    # print(df)
    params_names = df.columns.tolist()[2:]
    df_data = df[params_names]
    df_target = df['target']
    df_data = df_data[:10]
    df_target = df_target[:10]
    print('df_data = \n' + str(df_data))
    print('target = \n' + str(df_target))
    return df_data, df_target


def feature_selected_K(df_data, df_target, k):
    global params_names
    # ----------------------- 数据处理：标准化 ---------------------
    from sklearn.preprocessing import StandardScaler
    # 标准化，返回值为标准化后的数据
    df_stand_data = pd.DataFrame(StandardScaler().fit_transform(df_data), columns=params_names)
    np_target = np.array(df_target.values, dtype=np.float64)
    print('target.shape = ' + str(np_target.shape))
    np_stand_data = np.array(df_stand_data.values, dtype=np.float64)
    # data.dtype = np.float64# dataframe 转换成 ndarray
    print('data.shape = ' + str(np_stand_data[:, 0].shape))

    # ----------------------- 特征选择：相关系数法 ---------------------
    from scipy.stats import pearsonr
    import math
    # 选择K个最好的特征，返回选择特征后的数据
    # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
    # 参数k为选择的特征个数s
    y = np_target
    dict_corr = {}
    # 计算所有特征与目标值之间的pearson系数
    for col in range(np_stand_data.shape[1]):
        x = np_stand_data[:, col]
        pearson = list(pearsonr(x, y))
        print("name = " + str(params_names[col]) + " pearsonr", list(pearsonr(x, y)),
              ', score = ' + str(math.fabs(pearson[0])))
        # score 取绝对值，接近1正相关，接近-1负相关，0表示不相关
        dict_corr[params_names[col]] = math.fabs(pearson[0])
    print(dict_corr)

    # ------------------- 输出：最重要（pearson相关系数绝对值最高的）的前K个参数 -------------------------
    # 按字典集合中，每一个元组的第二个元素排列。
    d_order = sorted(dict_corr.items(), key=lambda x: x[1], reverse=True)
    print(d_order)
    # 获取score得分最高的前n个重要参数
    keys = []
    d_firstK = d_order[:k]
    for item in d_firstK:
        keys.append(item[0])
    print('前 ' + str(k) + ' 个重要参数名称 = ' + str(keys))
    # print(df_data[keys])
    return keys

if __name__ == '__main__':
    df_data, df_target = csvTodf()
    vitual_params = feature_selected_K(df_data, df_target, df_data.shape[1] - 10)
    print(vitual_params)