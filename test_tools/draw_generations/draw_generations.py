#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：test_tools
@File ：draw_generations.py
@Author ：Yang
@Date ：2022/1/16 17:12 
'''
import matplotlib.pyplot as plt
import numpy as np
import  json
def draw_target(x, y, init_points, n_iter):
    # 画图
    plt.plot(x, -y, label='lhs_bo  init_points = ' + str(init_points) + ', n_iter = ' + str(n_iter))
    max = y.max()
    max_indx = y.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.xlabel('iterations')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig("./lhs_target.png")
    plt.show()

if __name__ == '__main__':
    logpath = './logs.json'
    init_points = 15
    n_iter = 30

    target = []
    interations = []
    inters = 0
    for line in open(logpath, 'r'):
        data = json.loads(line)
        target.append(data['target'])
        interations.append(inters)
        inters += 1
    x = np.array(interations)
    y = np.array(target)
    draw_target(x, y, init_points, n_iter)