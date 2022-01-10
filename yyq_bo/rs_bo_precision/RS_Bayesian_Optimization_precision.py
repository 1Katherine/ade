import array
import datetime
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import sys
import os
# 将lhs_bo目录放在path路径中 (rs_bo_precision/ 是当前文件的上级目录 的上级目录
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from precision_baye_scode import SequentialDomainReductionTransformer
from precision_baye_scode import BayesianOptimization
from precision_baye_scode import JSONLogger
from precision_baye_scode import Events
import warnings
import os
from os.path import join as pjoin
import json

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
        # print(conf)
    # print(i)
    y = model.predict(np.matrix([i]))[0]
    # print(y)

    return -y

'''
    画出优化过程中的target值的变化过程
'''
time = datetime.datetime.now()
def draw_target(bo):
    # 画图
    plt.plot(-bo.space.target, label='rs_bo  init_points = ' + str(init_points) + ', n_iter = ' + str(n_iter))
    max = bo._space.target.max()
    max_indx = bo._space.target.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.xlabel('n_iter')
    plt.ylabel('runtime')
    plt.legend()

    plt.savefig("./rs_searching_config/target - " + str(time.strftime( '%Y-%m-%d %H-%M-%S')) + ".png")
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
    precisions = {}  # 参数精度
    for conf in vital_params['vital_params']:
        if conf in confDict:
            d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
            # 用 d1 字典更新 d2 字典（防止参数范围表中有重名的配置参数行，被多次添加到字典中）
            pbounds.update(d1)
            precisions[conf] = confDict[conf]['pre']
        else:
            print(conf, '-----参数没有维护: ', '-----')

    print(pbounds)
    print(precisions)
    '''
        开始贝叶斯优化，传入pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    '''
    # 记录贝叶斯优化开始时间
    startTime = datetime.datetime.now()

    bounds_transformer = SequentialDomainReductionTransformer()

    '''
        新增属性precisions，形如 {'spark.memory.storageFraction': 0.01, 'spark.executor.cores': 1.0
                            , 'spark.executor.memory': 1.0, 'spark.executor.instances': 1.0}
        更新时间：2021/1/5  14:15
    '''
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        precisions=precisions,
        verbose=2,
        random_state=1,
        bounds_transformer=bounds_transformer
    )

    # 把贝叶斯优化结果保存在logs文件中
    logpath = './logs/logs - ' + str(time.strftime( '%Y-%m-%d %H-%M-%S')) + '.json'
    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    init_points = 10
    n_iter = 30
    # 可以在maximize中指定acq=['ucb','ei','poi']
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
    # optimizer.maximize(init_points=init_points, n_iter=n_iter)
    print('optimizer.max = ' + str(optimizer.max))
    draw_target(optimizer)
    # print(optimizer.space.bounds)

    # 记录贝叶斯优化结束时间
    endTime = datetime.datetime.now()
    # 计算算法的持续时间（以秒为单位）
    searchDuration = (endTime - startTime).seconds

    # 打开json文件追加内容 - optimizer.max
    # max = str(optimizer.max)
    # fr = open(logpath, 'a')
    # model=json.dumps(max)
    # fr.write(model)
    # fr.close()