import pandas as pd
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt
from sko.GA import GA
import shutil
import os
import time
import csv

# 服务器运行spark时config文件
config_run_path = "/usr/local/home/zwr/hibench-spark-config/wordcount-100G-ga/"
# 重要参数
vital_params_path = "/usr/local/home/dzc/parameters_set.txt"
# 维护的参数-范围表
conf_range_table = "/usr/local/home/dzc/Spark_conf_range.xlsx"
# 保存配置
generation_confs = "/usr/local/home/dzc/generationBestConf.csv"
# 保存 GA的参数
ga_confs_path = "/usr/local/home/dzc/gaConfs.csv"
# 保存所有的 Y
all_history_Y_save_path = '/usr/local/home/dzc/all_history_y.csv'

# 格式化参数配置：精度、单位等
def formatConf(conf, value):
    res = ''
    # 处理精度
    if confDict[conf]['pre'] == 1:
        res = round(value)
    elif confDict[conf]['pre'] == 0.01:
        res = round(value, 2)
    # 添加单位
    if not pd.isna(confDict[conf]['unit']):
        # 布尔值
        if confDict[conf]['unit'] == 'flag':
            res = str(bool(res)).lower()
        # 列表形式的参数（spark.serializer、spark.io.compression.codec等）
        elif confDict[conf]['unit'] == 'list':
            rangeList = confDict[conf]['Range'].split(' ')
            res = rangeList[int(res)]
        # 拼接上单位
        else:
            res = str(res) + confDict[conf]['unit']
    else:
        res = str(res)
    return res

# 1、单个配置 p写入到 /usr/local/home/zwr/hibench-spark-config/wordcount-100G-ga   命名：config1
# 2、run获取执行时间并返回
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
        rumtime = 100000.0
    else:
        last_runtime = runtime
    return runtime
# 定义遗传算法的适应度函数
# 1、实际运行
configNum = 1
def schafferRun(p):
    global configNum
    # 打开配置文件模板
    fTemp = open('configTemp', 'r')
    # 复制模板，并追加配置
    fNew = open(config_run_path + 'config' + str(configNum), 'a+')
    shutil.copyfileobj(fTemp, fNew, length=1024)
    try:
        for i in range(p.size):
            fNew.write(' ')
            fNew.write(vital_params['vital_params'][i])
            fNew.write('\t')
            fNew.write(formatConf(vital_params['vital_params'][i], p[i]))
            fNew.write('\n')
    finally:
        fNew.close()
    runtime = run(configNum)
    configNum += 1
    return runtime


# 读取重要参数
vital_params = pd.read_csv(vital_params_path)
# 参数范围和精度，从参数范围表里面获取
sparkConfRangeDf = pd.read_excel(conf_range_table)
sparkConfRangeDf.set_index('SparkConf', inplace=True)
confDict = sparkConfRangeDf.to_dict('index')
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
        print('-----该参数没有维护: ', conf, '-----')
# 确定其他参数
fitFunc = schafferRun  # 适应度函数
nDim = len(vital_params)  # 参数个数
sizePop = 30    # 种群数量
maxIter = 10    # 迭代次数
probMut = 0.01  # 变异概率
# 调用遗传算法，记录整个搜索时间
startTime = datetime.datetime.now()
ga = GA(func=fitFunc, n_dim=nDim, size_pop=sizePop, max_iter=maxIter, prob_mut=probMut, lb=confLb, ub=confUb,
        precision=precisions)
best_x, best_y = ga.run()
endTime = datetime.datetime.now()
searchDuration = (endTime - startTime).seconds

# 存储参数配置
headers = ['func', 'n_dim', 'size_pop', 'max_iter', 'prob_mut', 'lb', 'ub', 'precision', 'searchDuration']
dicts = [{
    'func': fitFunc, 'n_dim': nDim, 'size_pop': sizePop, 'max_iter': maxIter,
    'prob_mut': probMut, 'lb': confLb, 'ub': confUb, 'precision': precisions,
    'searchDuration': searchDuration
}]
with open(ga_confs_path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(dicts)
# 存储每一代的最优解及其结果
generation_best_X = pd.DataFrame(ga.generation_best_X)
generation_best_X.columns = vital_params["vital_params"]
generation_best_X['runtime'] = ga.generation_best_Y
generation_best_X.to_csv(generation_confs, index=False)
# 存储所有搜索历史结果
pd.DataFrame(ga.all_history_Y).to_csv(all_history_Y_save_path)
