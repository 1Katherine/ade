import pandas as pd
import datetime
import matplotlib.pyplot as plt
import shutil
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rs_bayes_opt_newei.logger import JSONLogger
from rs_bayes_opt_newei.event import Events
from rs_bayes_opt_newei import BayesianOptimization
from rs_bayes_opt_newei import SequentialDomainReductionTransformer

import argparse
parser = argparse.ArgumentParser(description="parameters")
parser.add_argument('--benchmark', type=str, help='benchmark type')
parser.add_argument('--niters', type=int, default = 15, help='The number of iterations of the Bayesian optimization algorithm')
parser.add_argument('--ninit', type=int, default = 6, help='The number of initsample of the Bayesian optimization algorithm')
opts = parser.parse_args()
print('benchmark=' + str(opts.benchmark) + ', niters = ' + str(opts.niters) + ', ninit = ' + str(opts.ninit))


# 服务器运行spark时config文件
config_run_path = "/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI/config/" + opts.benchmark +  "/"
# 重要参数
vital_params_path = "/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI/parameters_set.txt"
# 维护的参数-范围表
conf_range_table = "/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI/Spark_conf_range_"  + opts.benchmark.split('-')[0] +  ".xlsx"
# 保存配置
generation_confs = "/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI/generationConf.csv"


def draw_target(bo):
    # 画图
    plt.plot(-bo.space.target, label='rs_bo_newEI  init_points = ' + str(init_points) + ', n_iter = ' + str(n_iter))
    max = bo._space.target.max()
    max_indx = bo._space.target.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.annotate('maxIndex:' + str(max_indx + 1), xy=(max_indx, -max), xycoords='data', xytext=(+20, +20),
                 textcoords='offset points'
                 , fontsize=12, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.annotate(str(round(-max, 2)) + 's', xy=(max_indx, -max), xycoords='data', xytext=(+20, -20),
                 textcoords='offset points'
                 , fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.xlabel('iterations')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig("/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI/target.png")
    plt.show()


def black_box_function(**params):
    i=[]
    for conf in vital_params['vital_params']:
        i.append(params[conf])
        # print(key)
    # print(i)
    # print(y)

    return -schafferRun(i)


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
    run_cmd = '/usr/local/home/zwr/' + opts.benchmark + '-ga.sh ' + str(configNum) + ' /usr/local/home/yyq/bo/rs_bo/rs_bo_newEI/config/'+ opts.benchmark
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
        for i in range(len(p)):
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
print(vital_params)

# 参数范围和精度，从参数范围表里面获取
sparkConfRangeDf = pd.read_excel(conf_range_table)
sparkConfRangeDf.set_index('SparkConf', inplace=True)
confDict = sparkConfRangeDf.to_dict('index')


    # 遍历训练数据中的参数，读取其对应的参数空间
d1={}
d2={}
precisions = []  # 参数精度
for conf in vital_params['vital_params']:
    if conf in confDict:
        d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
        d2.update(d1)
        precisions.append(confDict[conf]['pre'])
    else:
        print(conf,'-----参数没有维护: ', '-----')

#print(d2)
# 确定其他参数
startTime = datetime.datetime.now()



bounds_transformer = SequentialDomainReductionTransformer()

#定义贝叶斯优化模型 
optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=d2,
        verbose=2,
        random_state=1,
        # bounds_transformer=bounds_transformer
    )
logpath = "./logs.json"
logger = JSONLogger(path=logpath)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

init_points = opts.ninit
n_iter = opts.niters
optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
print(optimizer.max)
draw_target(optimizer)

# 存储数据
# 读取json文件, 转成csv
import json

res_df = pd.DataFrame()
for line in open(logpath).readlines():
    one_res = {}
    js_l = json.loads(line)
    one_res['target'] = -js_l['target']
    for pname in sorted(d2):
        one_res[pname] = js_l['params'][pname]
    df = pd.DataFrame(one_res, index=[0])
    res_df = res_df.append(df)
# 设置索引从1开始
res_df.index = range(1, len(res_df) + 1)
res_df.to_csv(generation_confs)