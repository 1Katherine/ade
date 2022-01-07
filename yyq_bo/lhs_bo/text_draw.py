#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yyq_bo 
@File ：text_draw.py
@Author ：Yang
@Date ：2022/1/7 16:00 
'''
import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.plot([1,2,3],[4,5,6])
# ax.text(1,4,'best',fontsize=14)
x = 1
y = 4
ax.annotate('best',xy = (x,y),xytext = (x,y+1), fontsize=14, horizontalalignment='center', arrowprops =dict(arrowstyle='->'))
plt.show()