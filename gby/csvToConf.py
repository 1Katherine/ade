'''
将GA算法每一代最优解保存下来的 csv 转为实际的 config配置
'''
import pandas as pd
import shutil
import random
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, normalize

name='xgb'

df = pd.read_csv('./searching_config/'+name+'generationbestConf.csv')
df = df.drop('runtime', 1)
sparkConfRangeDf = pd.read_excel('Spark_conf_range.xlsx')
sparkConfRangeDf.set_index('SparkConf', inplace=True)
confDict = sparkConfRangeDf.to_dict('index')





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
            res = rangeList[res]
        # 拼接上单位
        else:
            res = str(res) + confDict[conf]['unit']
    else:
        res = str(res)
    return res


# 遍历csv每一组配置生成一个配置文件
print('-----写入配置开始-----')
for index, row in df.iterrows():
    # 打开配置文件模板
    fTemp = open('configTemp', 'r')
    # 复制模板，并追加配置
    fNew = open('./config/'+name+'/config' + str(index),'w')
    shutil.copyfileobj(fTemp, fNew, length=1024)
    try:
        for i, item in row.items():
            fNew.write(' ')
            fNew.write(i)
            fNew.write('\t')
            fNew.write(formatConf(i, item))
            fNew.write('\n')
    finally:
        fNew.close()
print('-----写入配置完成-----')
