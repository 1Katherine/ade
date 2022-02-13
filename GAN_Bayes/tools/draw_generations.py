#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tools 
@File ：draw_generations.py
@Author ：Yang
@Date ：2022/1/16 17:12 
'''
import matplotlib.pyplot as plt
import numpy as np
import  json
def draw_target(x, y, init_points, n_iter):
    # 画图
    plt.plot(x, -y, label=str(boname) + '  init_points = ' + str(init_points) + ', n_iter = ' + str(n_iter))
    max = y.max()
    max_indx = y.argmax()+1
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.annotate(str(-max) + 's', xy=(max_indx, -max), xycoords='data', xytext=(+20, -20), textcoords='offset points'
                 , fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.xlabel('iterations')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig("./target.png")
    plt.show()

if __name__ == '__main__':
    boname = 'ganrs'
    logpath = './logs.json'
    init_points = 8
    n_iter = 29

    target = []
    interations = []
    inters = 0
    for row, line in enumerate(open(logpath, 'r')):
        if row < 28:
            print(row)
            print(line)
            data = json.loads(line)
            target.append(data['target'])
            interations.append(inters + 1)
            inters += 1
    x = np.array(interations)
    y = np.array(target)
    draw_target(x, y, init_points, n_iter)