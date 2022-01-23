#  -*- coding: UTF-8 -*
import pandas as pd
import shutil
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from bayes_scode import JSONLogger, Events, BayesianOptimization,SequentialDomainReductionTransformer
import matplotlib.pyplot as plt
# 调用代码：python ganrs_Bayesian_Optimization_server.py --sampleType=all --ganrsGroup=4 --niters=10 --initFile=/usr/local/home/yyq/bo/ganrs_bo/wordcount-100G-GAN.csv
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
# 采样方式：0表示所有样本，1表示前ganrsGroup*2个样本，2表示间隔采样每一组样本中只选一个rs一个gan,3表示选择执行时间最少的几个样本作为初始样本
parser.add_argument('--sampleType', type=str, default = all,
                    help='all: for all samole, '
                         'firsttwogroup: The first two groups of random samples and GAN samples are used as initial samples, '
                         'interval: interval sampling, '
                         'best: 10 samples with the least execution time')
# 一组rs+gan样本数，比如2个rs2个gan，反复循环，则一组样本数为2+2=4
parser.add_argument('--ganrsGroup', type=int, default = 0, help='A set of random samples and the number of GAN samples.'
                                                                'For example, two random samples are followed by two GAN samples, so ganrsGroup is equal to 4')
parser.add_argument('--niters', type=int, default = 15, help='The number of iterations of the Bayesian optimization algorithm')
parser.add_argument('--initFile', type=str, default=None, help='50% random sampling and 50% GAN generated initial sample files')
args = parser.parse_args()
if args.ganrsGroup == 0:
    raise Exception("必须执行一组gan和rs的个数，比如每3个rs会有3个gan，--ganrsGroup=6")
if args.initFile == None:
    raise Exception("必须指定50%随机采样和50%GAN的初始样本文件")
print('--sampleType = ' + args.sampleType + '\t --ganrsGroup = '
      + str(args.ganrsGroup) + '\t --niters = ' + str(args.niters) + '\t --initFile = ' + args.initFile)


# 服务器运行spark时config文件
config_run_path = "/usr/local/home/yyq/bo/ganrs_bo/config/wordcount-100G/"
# 重要参数
vital_params_path = "/usr/local/home/yyq/bo/ganrs_bo/parameters_set.txt"
# 维护的参数-范围表
conf_range_table = "/usr/local/home/yyq/bo/ganrs_bo/Spark_conf_range_wordcount.xlsx"
# 保存配置
generation_confs = "/usr/local/home/yyq/bo/ganrs_bo/generationConf.csv"
# 采样方式有三种（0所有样本或者even/odd，1前ganrs_group*2个样本，2间隔ganrs_group // 2个样本采样，一组样本中只采样1个rs2个gan）
sample_type = args.sampleType
# 一组rs+gan的样本数
ganrs_group = args.ganrsGroup
# 选择前headn个样本采样(前两组配置）
headn = ganrs_group * 2
# 间隔ganrs_interval个样本采样
ganrs_interval = ganrs_group // 2

# --------------------- 生成 gan-rs 初始种群 start -------------------
initpoint_path = args.initFile
initsamples_df = pd.read_csv(initpoint_path)

# 取所有样本作为bo初始样本
def ganrs_samples_all():
    # 初始样本
    initsamples = initsamples_df[vital_params_list].to_numpy()
    print('选择50%rs和50%gan的所有样本作为bo算法的初始样本,样本个数为:' + str(len(initsamples)))
    return initsamples

# 获取dataframe的前n行样本作为初始样本
def get_head_n(n):
    print('取出前' + str(n) + '个样本')
    initsamples_head = initsamples_df.head(n)
    initsamples = initsamples_head[vital_params_list].to_numpy()
    print('取出前两组样本作为初始样本：, shape = ' + str(initsamples.shape))
    return initsamples

# 每隔n行取一行
def get_ganrs_intevaln(n):
    a = []
    for i in range(0, len(initsamples_df), n):  ##每隔86行取数据
        a.append(i)
    sample = initsamples_df.iloc[a]
    initsamples = sample[vital_params_list].to_numpy()
    print('间隔采样，取出的行号为：' + str(a) + ' , shape = ' + str(initsamples.shape))
    return initsamples

# 样本按照runtime 升序排序，获取runtime最少的前n个样本作为初始样本
def get_best_n(n):
    initsamples_sort = initsamples_df.sort_values(by='runtime', ascending=True)
    initsamples_head = initsamples_sort[vital_params_list].head(n)
    initsamples = initsamples_head.to_numpy()
    print('把执行时间最少的前几个样本作为初始样本，shape=' + str(initsamples.shape))
    return initsamples
# --------------------- 生成 gan-rs 初始种群 end -------------------


def black_box_function(**params):
    i=[]
    for conf in vital_params['vital_params']:
        i.append(params[conf])
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


def draw_target(bo):
    # 画图
    plt.plot(-bo.space.target, label='ganrs_bo  init_points = ' + str(init_points) + ', n_iter = ' + str(n_iter))
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
    plt.savefig("/usr/local/home/yyq/bo/ganrs_bo/ganrs_target.png")
    plt.show()


if __name__ == '__main__':
    # 读取重要参数
    vital_params = pd.read_csv(vital_params_path)
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
    # 按照贝叶斯优化中的key顺序,得到重要参数的名称vital_params_name用于把json结果文件转成dataframe存成csv，以及重要参数+执行时间列vital_params_list用于读取初始样本
    vital_params_name = sorted(d2)
    vital_params_list = sorted(d2)
    vital_params_list.append('runtime')
    # ------------------ 选择初始样本（3个方法选其一） start -------------
    if sample_type == 'all':
        # 选择所有样本
        initsamples = ganrs_samples_all()
    elif sample_type == 'firsttwogroup':
        # 选择前n个样本
        initsamples = get_head_n(n=headn)
    elif sample_type == 'interval':
        # 每隔3个样本选择一个样本（包括第三个样本）
        initsamples = get_ganrs_intevaln(n = ganrs_interval)
    elif sample_type == 'best':
        initsamples = get_best_n(n=8)
    else:
        raise Exception("[!] 请在all、firsttwogroup、interval、best中选择一种初始样本方式，firsttwogroup表示前两组样本，"
                        "interval表示间隔采样，best表示执行时间最少前几个样本")
    # ------------------ 选择初始样本（3个方法选其一） end -------------

    bounds_transformer = SequentialDomainReductionTransformer()
    #定义贝叶斯优化模型
    optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=d2,
            verbose=2,
            random_state=1,
            bounds_transformer=bounds_transformer,
            custom_initsamples=initsamples
        )
    logpath = "/usr/local/home/yyq/bo/ganrs_bo/logs.json"
    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    init_points = len(initsamples)
    n_iter = args.niters
    print('inerations：' + str(n_iter))
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
    print(optimizer.max)
    draw_target(optimizer)


    #存储数据
    # 读取json文件, 转成csv
    import json
    res_df = pd.DataFrame()
    for line in open(logpath).readlines():
        one_res = {}
        js_l = json.loads(line)
        one_res['target'] = -js_l['target']
        for pname in vital_params_name:
            one_res[pname] = js_l['params'][pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df)+1)
    res_df.to_csv(generation_confs)
