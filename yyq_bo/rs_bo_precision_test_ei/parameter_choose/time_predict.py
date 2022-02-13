#2021/9/26
#用于构造执行时间预测模型
import argparse
import datetime
import random
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, normalize
import xgboost as xgb


import time


class Model_construct:
    def __init__(self, name, features, data, save_path, target):
        self.name = name
        self.final_features = []  # 最终选出的特征
        self.final_importance = []  # 最终选出特征的重要性
        self.min_error = 1.0  # 最小误差，最终选择的特征就是误差最低点的特征

        self.model = lgb.LGBMRegressor()
        self.data = data
        self.feature_length = len(features)

        self.init_features = []
        self.init_features = features

        self.save_path = str(save_path)  # 生成文件存储路径


        self.target = target


        print("训练数据行数" + str(len(self.data)))
        # print(len(self.data))

    # 根据名称构建模型
    def build_model(self):
        if self.name.lower() == "lgb":
            model =lgb.LGBMRegressor()
        elif self.name.lower() == "gdbt":
            model = GradientBoostingRegressor()
        elif self.name.lower() =="ada":
            model = AdaBoostRegressor()
        elif self.name.lower() =="xgb":
            model = xgb.XGBRegressor()
        else:
            model= RandomForestRegressor()

        return model

    # 切分数据做标准化
    def process_data(self, features):
        # 切分数据集,测试集占0.25

        features_data = self.data[features]
        target_data = self.data[self.target]

        x_train, x_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.20, random_state=22)

        # 做标准化`
        # transfer = StandardScaler()
        # x_train = transfer.fit_transform(x_train)
        # x_test = transfer.transform(x_test)
        return x_train, x_test, y_train, y_test

    # 计算误差
    def error_calculate(self, y_predict, y_test):
        y_test = y_test.tolist()
        test_length = len(y_test)
        error_percentage = 0
        # print(test_length)

        for i in range(0, test_length):
            # print(y_predict[i])
            # print((abs(y_test[i] - y_predict[i]) / y_predict[i]))
            error_percentage = error_percentage + (abs(y_test[i] - y_predict[i]) / y_test[i])

        # 所有误差取平均值

        error_percentage = error_percentage / test_length

        return error_percentage

    def main(self):
        startTime = datetime.datetime.now()
        print("开始训练")
        print(startTime)
        print("\n")
        self.train(self.init_features)
        endTime = datetime.datetime.now()
        duration = (endTime - startTime).seconds
        minutes_ = duration // 60
        seconds_ = duration - minutes_ * 60
        print("结束训练")
        print(endTime)
        print("\n")
        str_time = str(minutes_) + "m" + str(seconds_) + "s"
        print("训练时间" + str_time)
        print("\n")


    def train(self, features):
        # 取训练集，测试集
        x_train, x_test, y_train, y_test = self.process_data(features)

        # 构建模型
        model = self.build_model()
        # 训练
        model.fit(x_train, y_train)
        # 预测
        y_predict = model.predict(x_test)
        # print(y_predict)
        # print(type(y_predict))
        # print(type(y_test))

        # 记录特征重要性
        features_importance = model.feature_importances_

        # 计算误差
        error_percentage = self.error_calculate(y_predict, y_test)
        error_percentage = round(error_percentage, 3)
        print(error_percentage)
        self.min_error= error_percentage
        fileObject = open(self.save_path+self.name + "_error.txt", 'a+')
        fileObject.write(str(error_percentage))
        joblib.dump(model,self.save_path+"time_predict_"+self.name+".pkl")










