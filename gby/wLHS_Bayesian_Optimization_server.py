import datetime
import os
import time
import shutil
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib
from bo_scode_wlhs import SequentialDomainReductionTransformer
from bo_scode_wlhs import BayesianOptimization
from bo_scode_wlhs.logger import JSONLogger
from bo_scode_wlhs.event import Events
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

'''
    贝叶斯的黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
    1. 贝叶斯先生成一个样本
    2. 根据黑盒模型获取执行时间（通过将样本传给
'''
def black_box_function(**params):
    paras=[]
    for conf in vital_params['vital_params']:
        paras.append(params[conf])
        # print(key)
    return -schafferRun(paras)


'''
    传入一个样本paras,生成对应的配置文件并运行，得到执行时间
'''
# 服务器运行spark时config文件
configNum = 1
def schafferRun(paras):
    global configNum
    # 打开配置文件模板
    fTemp = open('configTemp', 'r')
    # 复制模板，打开 config文件，并追加配置
    fNew = open(config_run_path + 'config' + str(configNum), 'a+')
    shutil.copyfileobj(fTemp, fNew, length=1024)
    try:
        for p in range(len(paras)):
            fNew.write(' ')
            fNew.write(vital_params['vital_params'][p])
            fNew.write('\t')
            fNew.write(formatConf(vital_params['vital_params'][p], paras[p]))
            fNew.write('\n')
    finally:
        fNew.close()
    runtime = run(configNum)
    configNum += 1
    return runtime

'''
    1、单个配置 p写入到 /usr/local/home/yyq/wlhs_bo/config/wordcount-100G/   命名：config1
    2、run获取执行时间并返回
'''
last_runtime = 1.0
def run(configNum):
    # configNum = None
    # 使用给定配置运行spark
    run_cmd = '/usr/local/home/zwr/wordcount-100G-ga.sh ' + str(configNum)
    os.system(run_cmd)
    # 睡眠3秒，保证hibench.report文件完成更新后再读取运行时间
    time.sleep(3)
    # 获取此次spark程序的运行时间
    get_time_cmd = 'tail -n 1 /usr/local/home/hibench/hibench/report/hibench.report'
    line = os.popen(get_time_cmd).read()
    runtime = float(line.split()[4])
    global last_runtime
    if runtime == last_runtime:
        runtime = 100000.0
    else:
        last_runtime = runtime
    return runtime



'''
    将我们项目的参数范围文件转换为pbounds格式，传入给采样算法抽样
'''
def rangecsv_to_pbounds(vital_params, confDict):
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
    return pbounds, precisions


'''
    画出优化过程中的target值的变化过程
'''
def draw_target(bo):
    # 画图
    plt.plot(-bo.space.target, label='bo')
    plt.xlabel('迭代次数')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig("./wlhs_searching_config/target.png")
    # plt.show()

'''
    格式化参数配置：精度、单位等（在适应度函数中使用，随机生成的参数将其格式化后放入配置文件中实际运行，记录执行时间）
    遗传算法随机生成的值都是实数值，根据配置对应range表格中的精度，将值处理为配置参数可以运行的值
'''
def formatConf(conf, value):
    res = ''
    # 1. 处理精度
    # s、m、g、M、flag、list设置的精度都是1
    if confDict[conf]['pre'] == 1:
        # round():返回浮点数x的四舍五入值
        res = round(value)
    # 浮点型设置的精度是0.01
    elif confDict[conf]['pre'] == 0.01:
        res = round(value, 2)
    # 2. 添加单位(处理带单位的数据、flag和list数据)
    if not pd.isna(confDict[conf]['unit']):
        # 布尔值 false\true
        if confDict[conf]['unit'] == 'flag':
            res = str(bool(res)).lower()
        # 列表形式的参数（spark.serializer、spark.io.compression.codec等）
        elif confDict[conf]['unit'] == 'list':
            rangeList = confDict[conf]['Range'].split(' ')
            # res = 1就获取列表中第二个值
            res = rangeList[int(res)]
        # 给数字添加单位
        else:
            res = str(res) + confDict[conf]['unit']
    else:
        res = str(res)
    return res








if __name__ == '__main__':
    # 服务器运行spark时config文件
    config_run_path = "/usr/local/home/yyq/bo/wlhs_bo/config/wordcount-100G/"
    # 重要参数
    vital_params_path = "/usr/local/home/yyq/bo/wlhs_bo/parameters_set.txt"
    # 维护的参数-范围表
    conf_range_table = "/usr/local/home/yyq/bo/wlhs_bo/Spark_conf_range_wordcount.xlsx"
    # 保存配置
    generation_confs = "/usr/local/home/yyq/bo/wlhs_bo/generationDomainConf.csv"

    # 读取重要参数
    vital_params = pd.read_csv(vital_params_path)

    # 参数范围和精度，从参数范围表里面获取
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    confDict = sparkConfRangeDf.to_dict('index')

    # 获取pbounds格式和精度 pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    pbounds, precisions = rangecsv_to_pbounds(vital_params, confDict)

    '''
        开始贝叶斯优化，传入pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    '''
    # 记录贝叶斯优化开始时间
    startTime = datetime.datetime.now()

    # bounds_transformer = SequentialDomainReductionTransformer()

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
        # bounds_transformer=bounds_transformer
    )


    # 1. 实例化一个 observer  对象
    logger = JSONLogger(path="./wlhs_searching_config/logs.json")
    # 2. 将观察者对象绑定到优化器触发的特定事件。
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    init_points = 30
    n_iter = 70
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    print('optimizer.max = ' + str(optimizer.max))
    # 画出target随迭代次数的变化
    draw_target(optimizer)

    # 记录贝叶斯优化结束时间
    endTime = datetime.datetime.now()
    # 计算算法的持续时间（以秒为单位）
    searchDuration = (endTime - startTime).seconds


    '''
        新建dataframe文件，用于保存贝叶斯优化中找到的最好的5个样本点
        
        1. 设置dataframe的列名为（all_参数名称，执行时间） 
    '''
    # index 存放贝叶斯优化过程中使用的所有参数名称 index = ['spark.default.parallelism', 'spark.executor.cores', 'spark.executor.instances', 'spark.executor.memory'
    index = []
    # 根据 optimizer 的第一个样本点获取 'params'中所有参数的名称
    for key, value in optimizer.res[0]['params'].items():
        index.append(key)


    data = pd.DataFrame(columns=[x for x in index])
    data['runtime'] = ''

    '''
        新建dataframe文件，用于保存贝叶斯优化的所有结果

        2.获取贝叶斯优化过程中搜索的所有样本点（总共 init_points + n_iter 个）存储到dataframe中
        3.dataframe按照runtime列排序，取前5行存入csv文件中  './searching_config/' + name + "generationbestConf.csv" 
    '''
    n = 0
    for result in optimizer.res:
        n = n + 1
        params_and_runtime = []
        runtime = -result['target']
        for key, value in result['params'].items():
            params_and_runtime.append(value)
        params_and_runtime.append(runtime)
        # print(paramter)
        data.loc[n] = params_and_runtime
    data.to_csv(generation_confs, index=False)

