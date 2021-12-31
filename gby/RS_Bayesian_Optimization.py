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
    画出优化过程中的target值的变化过程
'''
def draw_target(bo):
    def draw_target(bo):
        # 画图
        plt.plot(-bo.space.target, label='lhs_bo')
        max = bo._space.target.max()
        max_indx = bo._space.target.argmax()
        # 在图上描出执行时间最低点
        plt.scatter(max_indx, -max, s=20, color='r')
        plt.xlabel('迭代次数')
        plt.ylabel('runtime')
        plt.legend()
        # plt.savefig("./wlhs_searching_config/target.png")
        plt.show()

if __name__ == '__main__':
    name = 'rf'
    # 重要参数
    vital_params_path = './files100/' + name + "/selected_parameters.txt"
    print(vital_params_path)
    # 维护的参数-范围表
    conf_range_table = "Spark_conf_range_wordcount.xlsx"
    # 参数配置表（模型选出的最好配置参数）
    generation_confs = './searching_config/' + name + "generationbestConf.csv"

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

    init_points = 30
    n_iter = 60
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    print('optimizer.max = ' + str(optimizer.max))
    draw_target(optimizer)
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
