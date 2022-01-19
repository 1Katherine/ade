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
import matplotlib.pyplot as plt
import joblib
from bayes_scode import SequentialDomainReductionTransformer
from bayes_scode import BayesianOptimization
from bayes_scode import JSONLogger,Events
import warnings
'''
    对于已经生成的GANRS初始样本是包含了执行时间的，使用bo的搜索过程中不再实际运行初始样本，
    而是杨参数和对应的执行时间注册register到cache中，避免重复运行44个样本，增加耗时
'''
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
        model = joblib.load(modelfile + 'lgb/lgb.pkl')
    elif name.lower() == "gbdt":
        model = joblib.load(modelfile + 'gbdt/gbdt.pkl')
    elif name.lower() == "rf":
        model = joblib.load(modelfile + 'rf/rf.pkl')
    elif name.lower() == 'xgb':
        model = joblib.load(modelfile + 'xgb/xgb.pkl')
    elif name.lower() == 'ada':
        model = joblib.load(modelfile + 'ada/ada.pkl')
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
    # 画图
    plt.plot(-bo.space.target, label='rs_bo  init_points = ' + str(init_points))
    max = bo._space.target.max()
    max_indx = bo._space.target.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.xlabel('迭代次数')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig(generation_confs_png)
    plt.show()

# --------------------- 生成 gan-rs 初始种群 start -------------------
initpoint_path = './wordcount-100G-GAN.csv'
initsamples_df = pd.read_csv(initpoint_path)

def ganrs_samples_all():
    # 初始样本
    initsamples = initsamples_df[vital_params_list].to_numpy()
    return initsamples

def ganrs_samples_odd():
    initsamples_odd = initsamples_df[initsamples_df.index % 2 == 0]
    initsamples = initsamples_odd[vital_params_list].to_numpy()
    return initsamples

def ganrs_samples_even():
    initsamples_even = initsamples_df[initsamples_df.index % 2 == 1]
    initsamples = initsamples_even[vital_params_list].to_numpy()
    return initsamples

def get_ganrs_samples(kind):
    if kind == 'all':
        samples = ganrs_samples_all()
    elif kind == 'odd':
        samples = ganrs_samples_odd()
    elif kind == 'even':
        samples = ganrs_samples_even()
    else:
        raise Exception("[!] There is no option to get initsample ")
    return samples
# --------------------- 生成 gan-rs 初始种群 end -------------------

if __name__ == '__main__':
    # 生成初始样本数量：all表示所有的44个样本，even和odd表示只取奇数和偶数行
    sample_kind = 'all'
    name = 'lgb'
    modelfile = './files44/'
    init_points = 44
    n_iter = 15
    # 重要参数
    vital_params_path = modelfile + name + "/selected_parameters.txt"
    print(vital_params_path)
    # 维护的参数-范围表
    conf_range_table = "Spark_conf_range_wordcount.xlsx"
    # 参数配置表（模型选出的最好配置参数）
    generation_confs_csv = modelfile + 'result/' + name + " - generationbestConf - init_points=" + str(
        init_points) + " ,n_iter=" + str(n_iter) + ".csv"
    generation_confs_png = modelfile + 'result/pic/' + name + " - generationbestConf - init_points=" + str(
        init_points) + " ,n_iter=" + str(n_iter) + ".png"


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
    # 按照贝叶斯优化中的key顺序
    vital_params_list = sorted(pbounds)
    vital_params_list.append('runtime')
    # print('vital_params_list = ' + str(vital_params_list))
    # 生成初始样本
    initsamples = get_ganrs_samples(kind=sample_kind)
    # print(initsamples)

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
        bounds_transformer=bounds_transformer,
        custom_initsamples=initsamples
    )
    logpath = modelfile + 'result/' + name + " - generationbestConf - init_points=" + str(
        init_points) + " ,n_iter=" + str(n_iter) + ".json"
    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    print('optimizer.max = ' + str(optimizer.max))
    draw_target(optimizer)
