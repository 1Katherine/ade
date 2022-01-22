import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from bayes_scode import BayesianOptimization, SequentialDomainReductionTransformer, JSONLogger, Events
import warnings
# 调用代码：python server\ganrs_Bayesian_Optimization_server.py --kindSample=all/odd/even
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--kindSample', type=str, default = 'all')
args = parser.parse_args()
print('sample_kind = ' + str(args.kindSample))
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
    elif name.lower() == "gbdt":
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
    result_pic_path = modelfile + str(init_points) + '/result/pic/'
    if not os.path.exists(result_pic_path):
        os.makedirs(result_pic_path)
    generation_confs_png = result_pic_path + name + " - generationbestConf - init_points=" + str(
        init_points) + " ,n_iter=" + str(n_iter) + ".png"
    # 画图
    plt.plot(-bo.space.target, label='ganrs_bo  init_points = ' + str(init_points))
    max = bo._space.target.max()
    max_indx = bo._space.target.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.annotate(str(-max) + 's', xy=(max_indx, -max), xycoords='data', xytext=(+20, -20), textcoords='offset points'
                 , fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.xlabel('interations')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig(generation_confs_png)
    plt.show()

# --------------------- 生成 gan-rs 初始种群 start -------------------
'''
    使用方法：
        # 取所有样本/奇数行/偶数行 --- 奇偶数行仅针对2个rs后接2个gan样本的情况
        initsamples = get_ganrs_samples(kind=sample_kind)

        # 取前headn个数据 ---- 3个rs后接3个gan样本的情况可使用该方法
        ganrs_group = 6
        headn = ganrs_group * 2
        initsamples = get_head_n(n=headn)

        # 每隔ganrs_interval行取一个数据 ------ 3个rs后接3个gan样本的情况可使用该方法
        ganrs_group = 6
        ganrs_interval = ganrs_group // 2
        initsamples = get_ganrs_intevaln(n = ganrs_interval)
'''
initpoint_path = 'wordcount-100G-GAN-30.csv'
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

# 获取dataframe的前n行样本作为初始样本
def get_head_n(n):
    initsamples_head = initsamples_df.head(n)
    initsamples = initsamples_head[vital_params_list].to_numpy()
    return initsamples

# 每隔n行取一行
def get_ganrs_intevaln(n):
    a = []
    for i in range(0, len(initsamples_df), n):  ##每隔86行取数据
        a.append(i)
    print('取出的行号为：' + str(a))
    sample = initsamples_df.iloc[a]
    initsamples = sample[vital_params_list].to_numpy()
    return initsamples
# --------------------- 生成 gan-rs 初始种群 end -------------------

if __name__ == '__main__':
    # 生成初始样本数量：all表示所有的44个样本，even和odd表示只取奇数和偶数行
    sample_kind = args.kindSample
    name = 'gbdt'
    modelfile = './files30/'
    # 重要参数
    vital_params_path = modelfile + name + "/selected_parameters.txt"
    # 维护的参数-范围表
    conf_range_table = "E:\\Desktop\\github同步代码\\ade\\GAN_Bayes\\laptop\\Spark_conf_range_wordcount.xlsx"


    vital_params = pd.read_csv(vital_params_path)
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


    # 按照贝叶斯优化中的key顺序,得到重要参数的名称vital_params_name用于把json结果文件转成dataframe存成csv，以及重要参数+执行时间列vital_params_list用于读取初始样本
    vital_params_name = sorted(pbounds)
    vital_params_list = sorted(pbounds)
    vital_params_list.append('runtime')

    # 选择所有样本
    # initsamples = get_ganrs_samples(kind=sample_kind)

    # # 选择前12个样本
    # ganrs_group = 6
    # headn = ganrs_group * 2
    # initsamples = get_head_n(n=headn)

    # # 每隔3个样本选择一个样本（包括第三个样本）
    ganrs_group = 6
    ganrs_interval = ganrs_group // 2
    initsamples = get_ganrs_intevaln(n = ganrs_interval)

    init_points = len(initsamples)
    n_iter = 15
    result_path = modelfile + str(init_points) + '/result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    logpath = result_path + name + " - generationbestConf - init_points=" + str(
        init_points) + " ,n_iter=" + str(n_iter) + ".json"
    '''
        开始贝叶斯优化，传入pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    '''
    bounds_transformer = SequentialDomainReductionTransformer()
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
        bounds_transformer=bounds_transformer,
        custom_initsamples=initsamples
    )
    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
    print('optimizer.max = ' + str(optimizer.max))
    draw_target(optimizer)


    generation_confs_csv = modelfile + str(init_points) + '/result/' + name + " - generationbestConf - init_points=" + str(
        init_points) + " ,n_iter=" + str(n_iter) + ".csv"
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
    res_df.to_csv(generation_confs_csv)