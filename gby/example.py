import array
import datetime
import os
import time

import numpy as np
from bayes_opt import BayesianOptimization
import shutil
import random
import  csv
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, normalize
import joblib
from bayes_opt import SequentialDomainReductionTransformer

df=pd.read_csv('generationConf.csv')

df=df.sort_values('runtime').reset_index(drop=True)
df=df[1:11]
print(df)