#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ade 
@File ：test.py
@Author ：Yang
@Date ：2022/1/6 15:23 
'''
import sys
import os
# print(sys.path)
# # 父级目录加入系统路径
# sys.path.append("../../")
# print(sys.path)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(sys.path)

print(os.path.realpath(__file__))
print(os.path.dirname(os.path.realpath(__file__)))
# 当前文件的上级目录 的上级目录 = lhs_bo
print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))