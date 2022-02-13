# 读取最初的总数据(包含micro,os,container)，对三层分别进行特征选择

import parameter_choose  # 用于特征选择的类

# coding=UTF-8
import argparse
import datetime
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
import os

import time_predict

"""
parser=argparse.ArgumentParser()
parser.add_argument('-f','--filePath',help='Path of trainData')
parser.add_argument('-n','--name',help='name of algorithm')
parser.add_argument('-s','--save_path',help='path for saving files')
parser.add_argument('-t','--target',help='prediction target')
parser.add_argument('-step','--step_nums',help='the num of parameters droped each time')
parser.add_argument('-left','--left_nums',help='the num of parameters needed to be selected')

args=parser.parse_args()

filepath=args.filePath
name=args.name
save_path=args.save_path
target=str(args.target)
step=int(args.step_nums)
left_num=int(args.left_nums)
"""
fig = plt.figure()
fig.tight_layout()

filepath = "E:\ADE\python_file\parameter_choose\\train_data\\tpcds21G\case2\\tpcds-21G-sorting-parameters_runtime.csv"
# name = "ada"
save_path = "E:\ADE\python_file\parameter_choose\\train_data\\tpcds21G\case2\\phase1\\"
target = "runtime"
step = 1
left_num = 2


def get_data(file_path, name):
    data = pd.read_csv(file_path)
    all_columns = data.columns

    column_length = len(all_columns)

    print("\n")
    print(name + " 特征个数: ")
    print(str(column_length - 1))

    print(name + " 行数: ")
    print(str(len(data)))

    # 存放特征
    features_list = []
    for feature in all_columns:
        if feature != target:
            features_list.append(feature)
    return data, features_list


def plot_error_main(x_data, y_data, min_error_save_path, name):
    # x = [i for i in range(len(self.error_list))]
    # 画误差图，横坐标为剩余特征个数,纵坐标为误差

    plt.plot(x_data, y_data)
    plt.title("min error of each train")
    plt.xlabel("The rows of train data")
    plt.ylabel("MAPE")

    plt.tight_layout()
    plt.savefig(min_error_save_path + name + "_min_error.png")
    plt.clf()

    for error in y_data:
        print(error, " ")
    print("plot_error_main")
    print("\n")


global_min_error = 3
global_min_error_location = " "
global_min_error_location_eq = " "
# ["lgb", "rf", "ada", "gdbt", "xgb"]
flag = True  #true 表示训练第一阶段模型， false表示训练第二阶段模型
for name in [ "rf", "ada", "gdbt", "xgb"]:


    os.mkdir(save_path + name + "\\")
    min_error_list = []
    size_list = []
    for length in range(80, 126, 2):
        # 总数据
        data, features_list = get_data(file_path=filepath, name="parameters")

        new_save_path = save_path + name + "\\" + "size_" + str(length) + "\\"

        os.mkdir(new_save_path)

        if flag == True:
            total_choose = parameter_choose.Choose(name=name, features=features_list, step=step, prefix="parameters",
                                                   data=data[0:length],
                                                   save_path=new_save_path, target=target, left_num=left_num)
        else:
            total_choose = time_predict.Model_construct(name=name, features=features_list,
                                                        data=data[0:length],
                                                        save_path=new_save_path, target=target)
        total_choose.main()

        min_error_list.append(total_choose.min_error)
        size_list.append(length)

        print("total_choose.min_error" + str(total_choose.min_error))
        if global_min_error > total_choose.min_error:
            global_min_error = total_choose.min_error
            global_min_error_location = name + str(length) + str(global_min_error)

            print("global-----------------------" + str(global_min_error))

        if global_min_error == total_choose.min_error:
            global_min_error_location_eq = name + str(length) + str(global_min_error)

        sum_importance = 0
        for i in range(len(total_choose.final_features)):
            sum_importance += total_choose.final_importance[i]

        cur_sum = 0
        left_features = []
        for i in range(len(total_choose.final_importance)):
            cur_sum += total_choose.final_importance[i]
            left_features.append(total_choose.final_features[i])

        if flag==True:
            file = open(new_save_path + "selected_parameters.txt", mode="a+")
            file.write("vital_params")
            file.write("\n")
            for feature in left_features:
                file.write(feature)
                file.write("\n")
            file.close()

    plot_error_main(x_data=size_list, y_data=min_error_list, min_error_save_path=save_path + name + "\\", name=name)

file = open(save_path + "\\global_min_error.txt", mode="a+")
file.write(global_min_error_location)
file.write("\n")
file.write(global_min_error_location_eq)
file.write("\n")
file.close()

"""
interact_obj=micro_interact_new.Interaction_calculate(data=data,final_features=total_choose.final_features,
                                                      num=len(total_choose.final_features),
                                                      model=total_choose.model,
                                                      step_nums=20,save_path=save_path)

interact_obj.main()
"""
