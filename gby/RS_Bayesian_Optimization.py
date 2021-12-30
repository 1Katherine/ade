import array
import datetime
import os
import time

import numpy as np
import shutil
import random
import csv
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, normalize
import joblib
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings("ignore")

'''
    根据名称构建模型
'''


def build_model(name):
    if name.lower() == "lgb":
        model = lgb.LGBMRegressor()
    elif name.lower() == "gdbt":
        model = GradientBoostingRegressor()
    else:
        model = RandomForestRegressor()
    return model


'''
    不重新建模，使用已经构建好的模型
'''


def build_training_model(name):
    if name.lower() == "lgb":
        model = joblib.load('./files100/lgb/lgb.pkl')
    elif name.lower() == "gbdt":
        model = joblib.load('./files100/gbdt/gbdt.pkl')
    elif name.lower() == "rf":
        model = joblib.load('./files100/rf/rf.pkl')
    elif name.lower() == 'xgb':
        model = joblib.load('./files100/xgb/xgb.pkl')
    elif name.lower() == 'ada':
        model = joblib.load('./files100/ada/ada.pkl')
    else:
        raise Exception("[!] There is no option ")
    return model


'''
    贝叶斯的黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
'''


def black_box_function(**params):
    i = []
    model = build_training_model(name)
    for conf in vital_params['vital_params']:
        i.append(params[conf])
        # print(key)
    # print(i)
    y = model.predict(np.matrix([i]))[0]
    # print(y)

    return -y


'''
    切分数据，对切分的训练集和测试机做标准化
'''


def process_data(features):
    # 切分数据集,测试集占0.25
    features_data = data[features]
    target_data = data[target]
    x_train, x_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.25, random_state=22)

    # 做标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    return x_train, x_test, y_train, y_test


'''
    计算模型误差
    
'''


def error_calculate(y_predict, y_test):
    y_test = y_test.tolist()
    test_length = len(y_test)
    error_percentage = 0
    # print(test_length)

    for i in range(0, test_length):
        # print(y_predict[i])
        error_percentage = error_percentage + (abs(y_test[i] - y_predict[i]) / y_test[i])

        # 所有误差取平均值

    error_percentage = error_percentage / test_length

    return error_percentage


'''
    选择参数：构建模型，训练模型、预测y值，获取特征重要性，如果剩下的特征少于5个则保存模型退出，如果剩下特征大于5个则每次删除2个最不重要的特征递归直到退出
'''


def choose_features(features):
    final_features = []
    final_importance = []
    min_error = 1.0
    error_list = []

    # 取训练集，测试集
    x_train, x_test, y_train, y_test = process_data(features)
    # 构建模型
    model = build_model(name)
    # 训练
    model.fit(x_train, y_train)
    # 记录特征重要性
    features_importance = model.feature_importances_
    # 将特征和特征重要性拼到一起 格式如右 [('RM', 0.49359385750858875), ('LSTAT', 0.3256110013950264)]
    features_with_importance = list(zip(features, features_importance))
    # 根据特征重要性进行排序，component[1]为重要性
    # 按降序排序
    features_with_importance = sorted(features_with_importance, key=lambda component: component[1], reverse=True)

    # 预测
    y_predict = model.predict(x_test)
    # 计算误差
    error_percentage = error_calculate(y_predict, y_test)
    if min_error > error_percentage:
        min_error = error_percentage
        final_features = [x[0] for x in features_with_importance]
        final_importance = [x[1] for x in features_with_importance]
    error_list.append(error_percentage)

    # 格式化参数配置：精度、单位等
    # 直到剩下的特征数小于等于5停止
    # sum_importance = sum(x[1] for x in features_with_importance)
    if len(features_with_importance) > 5:
        # print("进入")
        f_length_ = len(features_with_importance)
        # 取除最不重要的2个
        features_with_importance = features_with_importance[0:f_length_ - 1]
        # 计算删除最不重要的特征后，新的特征重要性，以及特征重要性变化程度
        new_sum_importance = sum(x[1] for x in features_with_importance)
        # if (sum_importance-new_sum_importance)/sum_importance<0.05:
        # 计算剩下的新特征
        new_features = [x[0] for x in features_with_importance]
        # 用剩下的特征进行下一次训练
        choose_features(new_features)
    else:
        # 递归终止,并保存模型
        joblib.dump(model, name + ".pkl")
        # 输出最终选的特征
        print("features_with_importance: " + name, features)
        # 将最终选出的特征转为dataframe，并指定列名为 vital_params
        features = pd.DataFrame(features, columns=['vital_params'])
        # 将最终选出的特征保存到 model.name+'parameters_select.csv 文件
        pd.DataFrame.to_csv(features, name + 'parameters_select.csv', index=None)
        return


if __name__ == '__main__':
    name = 'rf'

    # 设置路径
    # 重要参数
    vital_params_path = './files100/' + name + "/selected_parameters.txt"
    print(vital_params_path)
    # 维护的参数-范围表
    conf_range_table = "Spark_conf_range_wordcount.xlsx"
    # 保存所有的 Y
    all_history_Y_save_path = 'all_history_y.csv'
    # 参数配置表（模型选出的最好配置参数）
    generation_confs = './searching_config/' + name + "generationbestConf.csv"
    # 读取数据（配置参数 + 执行时间（最后一列））
    data = pd.read_csv('data/wordcount-100G-sorting-parameters_runtime.csv')

    # 取出所有列属性
    all_columns = data.columns

    column_length = len(all_columns)
    # print(all_columns)
    # 取出特征(预测目标以外的所有列属性)
    features_global = all_columns[:column_length - 1]
    target = all_columns[-1]

    '''
        读取模型输出的重要参数
    '''
    vital_params = pd.read_csv(vital_params_path)
    print(vital_params)
    # 参数范围和精度，从参数范围表里面获取

    # 参数范围表
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    # SparkConf 列存放的是配置参数的名称
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    # 转化后的字典形式：{index(值): {column(列名): value(值)}}
    # {'spark.broadcast.blockSize': {'Range': '32-64m', 'min': 32.0, 'max': 64.0, 'pre': 1.0, 'unit': 'm'}
    confDict = sparkConfRangeDf.to_dict('index')

    '''
        获取pbounds格式 pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    '''
    # 遍历训练数据中的参数，读取其对应的参数空间
    d1 = {}
    pbounds = {}
    precisions = []  # 参数精度
    for conf in vital_params['vital_params']:
        if conf in confDict:
            d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
            # 用 d1 字典更新 d2 字典（防止参数范围表中有重名的配置参数行，被多次添加到字典中）
            pbounds.update(d1)
            precisions.append(confDict[conf]['pre'])
        else:
            print(conf, '-----参数没有维护: ', '-----')

    # print(pbounds)
    '''
        开始贝叶斯优化，传入pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    '''
    # 记录贝叶斯优化开始时间
    startTime = datetime.datetime.now()

    bounds_transformer = SequentialDomainReductionTransformer()

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
        # bounds_transformer=bounds_transformer
    )

    init_points = 60
    n_iter = 60
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    print('optimizer.max')
    print(optimizer.max)
    # print(optimizer.space.bounds)

    # 记录贝叶斯优化结束时间
    endTime = datetime.datetime.now()
    # 计算算法的持续时间（以秒为单位）
    searchDuration = (endTime - startTime).seconds

    # generation_best_X['runtime'] = ga.generation_best_Y
    # generation_best_X.to_csv(generation_confs, index=False)
    # print(data)

    '''
        新建dataframe文件，用于保存贝叶斯优化中找到的最好的5个样本点
        
        1. 设置dataframe的列名为（all_参数名称，执行时间） 
    '''
    # index 存放贝叶斯优化过程中使用的所有参数名称 index = ['spark.default.parallelism', 'spark.executor.cores', 'spark.executor.instances', 'spark.executor.memory'
    index = []
    # 根据 optimizer 的第一个样本点获取 'params'中所有参数的名称
    for key, value in optimizer.res[0]['params'].items():
        index.append(key)
    # print (index)

    data = pd.DataFrame(columns=[x for x in index])
    data['runtime'] = ''

    '''
        新建dataframe文件，用于保存贝叶斯优化中找到的最好的5个样本点

        2.获取贝叶斯优化过程中搜索的所有样本点（总共 init_points + n_iter 个）存储到dataframe中
        3.dataframe按照runtime列排序，取前5行存入csv文件中  './searching_config/' + name + "generationbestConf.csv" 
    '''
    # 获取最好的5个样本，存入 './rs_searching_config/' + name + " - generationbestConf"+ init_points + n_iter +".csv"  中
    generation_confs = './rs_searching_config/' + name + " - generationbestConf - init_points="+ str(init_points) + " ,n_iter=" + str(n_iter) +".csv"
    n = 0
    # result 总共 init_points + n_iter 个，遍历这些样本点
    for result in optimizer.res:
        n = n + 1
        params_and_runtime = []
        runtime = -result['target']
        # print(result['params'])
        # print(execution)
        # result['params'] 中 key是参数名称，value是采样的参数值
        for key, value in result['params'].items():
            # print(key)
            # print(value)
            params_and_runtime.append(value)
        params_and_runtime.append(runtime)
        # print(paramter)
        data.loc[n] = params_and_runtime
    # 按照执行时间排序
    data = data.sort_values('runtime').reset_index(drop=True)
    # 获得执行时间最短的前五个样本点
    data = data[:5]
    # 将执行时间最短的5个样本点存为 csv 文件
    data.to_csv(generation_confs, index=False)

    # pd.DataFrame(optimizer.res).to_csv(all_history_Y_save_path)
