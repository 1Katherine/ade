#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   exploration.py   
@Author ：Yang 
@CreateTime :   2022/2/8 13:04 
@Reference : 
'''

import os
import numpy as np

XI = np.linspace(0, 2, 5, endpoint=True)
for xi in XI :
    xi = round(xi,2)
    os.system('./run.sh')
    # os.system('python ganrs_Bayesian_Optimization_regitser.py --xi=' + str(xi))