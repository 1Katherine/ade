#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yyq_bo 
@File ：run_wlhs.py
@Author ：Yang
@Date ：2022/1/6 19:17 
'''
import os
# 获取当前文件所属的文件夹路径
# pwd = os.path.dirname(os.path.abspath(__file__))
# code = pwd + '\\wLHS_Bayesian_Optimization.py'
# print(code)
# os.system('python ' + code)

for i in range(20):
    os.system('python wLHS_Bayesian_Optimization.py')