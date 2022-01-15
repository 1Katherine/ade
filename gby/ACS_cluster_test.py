import datetime
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from sko.GA import GA



# 主机上运行的代码


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
    画出优化过程中的target值的变化过程
'''
def draw_target(bo):
    # 画图
    plt.plot(-bo.space.target, label='ga')
    max = bo._space.target.max()
    max_indx = bo._space.target.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.xlabel('iterations')
    plt.ylabel('runtime')
    plt.legend()
    time = datetime.datetime.now()
    plt.savefig("./rs_searching_config/target.png")
    plt.show()

'''
    黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
'''
def black_box_function(params):
    print(params)
    model = build_training_model(name)
    runtime = model.predict(np.matrix([params]))[0]
    return runtime

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
        获取格式
    '''
    # 遍历训练数据中的参数，读取其对应的参数空间
    confLb = []  # 参数空间上界
    confUb = []  # 参数空间下界
    precisions = []  # 参数精度
    for conf in vital_params['vital_params']:
        if conf in confDict:
            confLb.append(confDict[conf]['min'])
            confUb.append(confDict[conf]['max'])
            precisions.append(confDict[conf]['pre'])
        else:
            print(conf, '-----参数没有维护: ', '-----')

    # print(pbounds)
    '''
        开始遗传算法
    '''
    startTime = datetime.datetime.now()
    # 确定其他参数
    fitFunc = black_box_function  # 适应度函数
    nDim = len(vital_params)  # 参数个数
    sizePop = 30  # 种群数量
    maxIter = 10  # 迭代次数
    probMut = 0.01  # 变异概率
    ga = GA(func=fitFunc, n_dim=nDim, size_pop=sizePop, max_iter=maxIter, prob_mut=probMut, lb=confLb, ub=confUb,
            precision=precisions)
    best_x, best_y = ga.run()
    endTime = datetime.datetime.now()
    searchDuration = (endTime - startTime).seconds
    print('best_x : ' + str(best_x))
    print('best_y : ' + str(best_y))

    # %% Plot the result
    import pandas as pd
    import matplotlib.pyplot as plt

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()



