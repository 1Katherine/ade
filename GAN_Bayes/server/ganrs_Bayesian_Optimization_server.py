import pandas as pd
import shutil
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from bayes_scode import JSONLogger, Events, BayesianOptimization,SequentialDomainReductionTransformer
import matplotlib.pyplot as plt
# 调用代码：python server\ganrs_Bayesian_Optimization_server.py --kindSample=all/odd/even
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--kindSample', type=str, default = 'all')
args = parser.parse_args()
print('sample_kind = ' + str(args.kindSample))



# 服务器运行spark时config文件
config_run_path = "/usr/local/home/yyq/bo/ganrs_bo/config/wordcount-100G/"
# 重要参数
vital_params_path = "/usr/local/home/yyq/bo/ganrs_bo/parameters_set.txt"
# 维护的参数-范围表
conf_range_table = "/usr/local/home/yyq/bo/ganrs_bo/Spark_conf_range_wordcount.xlsx"
# 保存配置
generation_confs = "/usr/local/home/yyq/bo/ganrs_bo/generationConf.csv"

# --------------------- 生成 gan-rs 初始种群 start -------------------
initpoint_path = '/usr/local/home/yyq/bo/ganrs_bo/wordcount-100G-GAN-44.csv'
initsamples_df = pd.read_csv(initpoint_path)

# 取44个初始样本中所有样本作为bo初始样本
def ganrs_samples_all():
    # 初始样本
    initsamples = initsamples_df[vital_params_list].to_numpy()
    return initsamples

# 取44个初始样本中的偶数行样本作为bo初始样本
def ganrs_samples_odd():
    initsamples_odd = initsamples_df[initsamples_df.index % 2 == 0]
    initsamples = initsamples_odd[vital_params_list].to_numpy()
    return initsamples

# 取44个初始样本中的奇数行样本作为bo初始样本
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
    plt.annotate(str(-max) + 's', xy=(max_indx, -max), xycoords='data', xytext=(+20, -20), textcoords='offset points'
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
    # 生成初始样本
    sample_kind = args.kindSample
    initsamples = get_ganrs_samples(kind=sample_kind)

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
    n_iter = 15
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
