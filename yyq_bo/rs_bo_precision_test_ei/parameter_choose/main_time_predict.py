# coding=UTF-8
#构造
import argparse
import time
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
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import time_predict


parser=argparse.ArgumentParser()
parser.add_argument('-f','--filePath',help='Path of trainData')
parser.add_argument('-n','--name',help='name of algorithm')
parser.add_argument('-s','--save_path',help='path for saving files')
parser.add_argument('-t','--target',help='prediction target')

"""
args=parser.parse_args()

filepath=args.filePath
name=args.name
save_path=args.save_path
target=str(args.target)
"""

args=parser.parse_args()

class main_time_predict:
    def __init__(self, filepath, save_path):
        self.filepath = filepath
        self.name="lgb"
        self.save_path = save_path
        self.target="runtime"

    def get_data(self, file_path, name):
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
            if feature != self.target:
                features_list.append(feature)
        return data, features_list

    def main(self):
        # for name in ["lgb", "rf", "ada", "gdbt", "xgb"]:
        for name in ["rf", "ada", "gdbt", "xgb"]:
            if not os.path.isdir(self.save_path + name):
                os.makedirs(self.save_path + name)
            min_error_list = []
            size_list = []

            data, features_list = self.get_data(file_path=self.filepath, name="parameters")
            new_save_path = self.save_path + name + "\\"

            model_construct_Obj= time_predict.Model_construct(name=name, save_path=new_save_path, target=self.target, data=data, features=features_list)
            model_construct_Obj.main()

if __name__ == '__main__':
    filepath = "wordcount-100G-GAN-3+3.csv"
    save_path = "./model/" + str(time.strftime('%Y-%m-%d')) + '/' + str(time.strftime('%H-%M-%S')) + '/'
    predict_time_model = main_time_predict(filepath=filepath, save_path=save_path)
    predict_time_model.main()
