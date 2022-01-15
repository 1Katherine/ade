#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987


import numpy as np
from .base import SkoBase
from sko.tools import func_transformer
from abc import ABCMeta, abstractmethod
from .operators import crossover, mutation, ranking, selection


class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple()):
        self.func = func_transformer(func)
        assert size_pop % 2 == 0, 'size_pop must be even integer'
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(constraint_eq)  # a list of equal functions with ceq[i] = 0
        self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with c[i] <= 0

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

    @abstractmethod
    def chrom2x(self, Chrom):
        pass

    # 根据X 和 func 返回性能和y的值
    def x2y(self):
        print('-------------------- 开始 x2y(self) ----------------------')
        self.Y_raw = self.func(self.X)
        # 如果有约束
        if not self.has_constraint:
            self.Y = self.Y_raw
        # 如果没有约束
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        print('--------------------- 结束 x2y(self) ---------------------')
        return self.Y

    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()

            print('--------------------进行排序、选择、交叉、变异前的种群-------------------')
            print(str(self.Chrom))
            print('------------')

            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            print('--------------------进行排序、选择、交叉、变异后的种群-------------------')
            print(str(self.Chrom))
            print('------------')

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

    fit = run


class GA(GeneticAlgorithmBase):
    """genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint
    constraint_ueq : tuple
        unequal constraint
    precision : array_like
        The precision of every variables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py
    """

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq)
        # n_dim = 搜索参数的个数（比如有10个需要优化的重要参数 n_dim = 10）
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        print('lb = ' + str(lb))
        print('ub = ' + str(ub))
        # 返回一个指定形状和数据类型的新数组，并且数组中的值都为1   precision = [0.01, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array
        print('precision = ' + str(precision))

        # Lind is the num of genes of every variable of func（segments）
        # Lind是函数（segments）中每个变量的基因数量。
        # Lind_raw = [ 5.357552    1.5849625   2.32192809  2.32192809  5.357552    5.04439412    30.          8.23361968  5.61470984  8.94836723]
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        # ceiling向上取整，转换数组的数据类型为int
        # Lind = [ 6  2  3  3  6  6 31  9  6  9]
        self.Lind = np.ceil(Lind_raw).astype(int)

        # if precision is integer:
        # if Lind_raw is integer, which means the number of all possible value is 2**n, no need to modify
        # if Lind_raw is decimal, we need ub_extend to make the number equal to 2**n,
        # 如果精度是整数，基因数量也是整数，int_mode_ = 0 false
        # 如果精度是整数，基因数量不是整数，int_mode_ = 1 true
        # int_mode_ = [False  True  True  True False  True  True  True  True  True]
        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        # int_mode_为0，意味着所有可能的值的数量是2**n，不需要修改。
        # np.any 判断参数如果全为false则返回false，有一个true则返回true
        # int_mode = True
        self.int_mode = np.any(self.int_mode_)
        # int_mode_为1，我们需要ub_extend来使这个数字等于2**n
        if self.int_mode:
            # 如果 int_mode_ = true， ub_extend = self.lb + (np.exp2(self.Lind) - 1) * self.precision
            # 如果 int_mode_ = false， ub_extend = self.ub
            self.ub_extend = np.where(self.int_mode_
                                      , self.lb + (np.exp2(self.Lind) - 1) * self.precision
                                      , self.ub)
            print('self.ub_extend = ' + str(self.ub_extend))
        print('int_mode_ = ' + str(self.int_mode_))
        print('int_mode = ' + str(self.int_mode))
        # 染色体长度（所有变量的基因数量总和） len_chrom = 81
        self.len_chrom = sum(self.Lind)
        # 生成初始种群  Chrom = [[1 1 0... 0 1 1]
        self.crtbp()

    # 产生初始种群（0-1），行为种群个数，列为染色体长度（所有变量的基因数量总和）
    def crtbp(self):
        # create the population
        # 返回一个随机整数，范围从[0,2), 随机数的尺寸为 size_pop * len_chrom (sizePop * 81)
        # Chrom = [[1 1 0... 0 1 1]
        #          [0 0 0... 0 1 0]....
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        print('初始种群为 = ' + str(self.Chrom))
        return self.Chrom

    # 对每一个变量的基因片段生成 30个（size_pop）个数值 ，数值均为 0 - 1 之间的实数
    def gray2rv(self, gray_code):
        print('----------------------- 开始 gray2rv(self, gray_code) ----------------------')
        # gray_code 转为实值：整条染色体的一个片段
        # 输入是一个包含0和1的二维numpy数组。
        # 输出是一个一维的numpy数组，将输入的每一行转换成实数。
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        print('gray_code = ' + str(gray_code))
        # len_gray_code 为该变量对应的染色体数量
        _, len_gray_code = gray_code.shape
        # 按照列累加(第一列 = 第一列 ， 第二列 = 第一列+第二列 ， 第三列 = 第一列+第二列+第三列....)，除以2取余 = 0 / 1
        # b为 size = size_pop * len_gray_code
        b = gray_code.cumsum(axis=1) % 2
        print('b = ' + str(b))
        # np.logspace 对数等比数列 start=开始值，stop=结束值，num=元素个数，base=指定对数的底, endpoint=是否包含结束值
        # mask = 0.5**n (n = 1 .... len_gray_code) mask = [0.5      0.25     0.125    0.0625   0.03125  0.015625]
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        print('mask = ' + str(mask))
        # b * mask size = size_pop * len_gray_code
        print('b * mask : ' + str((b * mask)))
        # (b * mask).sum(axis=1) size = 1 * size_pop
        print('(b * mask).sum(axis=1) : ' + str((b * mask).sum(axis=1)))
        print('return of gray2rv(self, gray_code) : ' + str((b * mask).sum(axis=1) / mask.sum()))
        print('---------------------- 结束 gray2rv(self, gray_code) ------------------------')
        # 返回数组，个数为30（种群个数）
        return (b * mask).sum(axis=1) / mask.sum()


    def chrom2x(self, Chrom):
        print('----------------------- 开始 chrom2x(self, Chrom) -------------------------')
        # Lind = [6  2  3  3  6  6 31  9  6  9] 共有10个变量 cumsum计算一个数组各行的累加值
        cumsum_len_segment = self.Lind.cumsum()
        # cumsum_len_segment = [ 6  8 11 14 20 26 57 66 72 81]
        # X为一次迭代得到的size_pop个个体，矩阵行为size_pop，列为变量个数
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        # i = 0\1\2\3...9（变量个数） j = 6\8\11\14\....
        # 对每一个变量的片段染色体计算 gray2rv （一个变量对应 X 的多个列，有的对应2列 有的对应8列）
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                # 取初始种群 Chrom 的前6列 （第一个变量）
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                print('i = 0, Chrom_temp = ' + str(Chrom_temp))
            else:
                # 取 Chorm 的第 cumsum_len_segment[i - 1] 到第 cumsum_len_segment[i]列（其他变量列）
                # 取每一个变量对应的染色体子片段
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                print('i ！= 0, Chrom_temp = ' + str(Chrom_temp))
            # 将每一个变量的片段染色体的 0-1 之间的实数值放入返回值X的对应列中
            X[:, i] = self.gray2rv(Chrom_temp)
        print('X[:, i]' + str(X))

        # 如果int_mode = 1，精度为整数，基因数量不是整数 or 精度为小数，基因数量为整数
        # 将 X 为 0 - 1之间的实数放大为 lb - ub 之间
        if self.int_mode:
            # ub可能不遵从精度，这也是可以的。 例如，如果精度=2，lb=0，ub=5，那么x可以是5
            X = self.lb + (self.ub_extend - self.lb) * X
            # 如果值超过上界，则这个值等于上界，否则不变（维护参数边界）
            X = np.where(X > self.ub, self.ub, X)
            print('lb = ' + str(self.lb))
            print('ub = ' + str(self.ub))
            print('(self.ub - self.lb) = ' + str((self.ub - self.lb)))
            print('int_mode = 1, X = ' + str(X))
            # the ub may not obey precision, which is ok.
            # for example, if precision=2, lb=0, ub=5, then x can be 5
        # 如果int_mode = 0
        else:
            X = self.lb + (self.ub - self.lb) * X
            print('(self.ub - self.lb) = ' + str((self.ub - self.lb)))
            print('int_mode = 0, X = ' + str(X))
        print('------------------------- 结束 chrom2x(self, Chrom) -----------------------')
        return X

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

    def to(self, device):
        '''
        use pytorch to get parallel performance
        '''
        try:
            import torch
            from .operators_gpu import crossover_gpu, mutation_gpu, selection_gpu, ranking_gpu
        except:
            print('pytorch is needed')
            return self

        self.device = device
        self.Chrom = torch.tensor(self.Chrom, device=device, dtype=torch.int8)

        def chrom2x(self, Chrom):
            '''
            We do not intend to make all operators as tensor,
            because objective function is probably not for pytorch
            '''
            Chrom = Chrom.cpu().numpy()
            cumsum_len_segment = self.Lind.cumsum()
            X = np.zeros(shape=(self.size_pop, self.n_dim))
            for i, j in enumerate(cumsum_len_segment):
                if i == 0:
                    Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                else:
                    Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                X[:, i] = self.gray2rv(Chrom_temp)

            if self.int_mode:
                X = self.lb + (self.ub_extend - self.lb) * X
                X = np.where(X > self.ub, self.ub, X)
            else:
                X = self.lb + (self.ub - self.lb) * X
            return X

        self.register('mutation', mutation_gpu.mutation). \
            register('crossover', crossover_gpu.crossover_2point_bit). \
            register('chrom2x', chrom2x)

        return self



# 用遗传算法来解决TSP（旅行推销员问题）
class GA_TSP(GeneticAlgorithmBase):
    """
    Do genetic algorithm to solve the TSP (Travelling Salesman Problem)
    Parameters
    ----------------
    func : function
        The func you want to do optimal.
        It inputs a candidate solution(a routine), and return the costs of the routine.
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes corresponding to every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
    ```py
    num_points = 8
    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print('distance_matrix is: \n', distance_matrix)
    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    ```
    Do GA
    ```py
    from sko.GA import GA_TSP
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=8, pop=50, max_iter=200, Pm=0.001)
    best_points, best_distance = ga_tsp.run()
    ```
    """

    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.001):
        super().__init__(func, n_dim, size_pop=size_pop, max_iter=max_iter, prob_mut=prob_mut)
        self.has_constraint = False
        self.len_chrom = self.n_dim
        self.crtbp()

    def crtbp(self):
        # create the population
        tmp = np.random.rand(self.size_pop, self.len_chrom)
        self.Chrom = tmp.argsort(axis=1)
        return self.Chrom

    def chrom2x(self, Chrom):
        return Chrom

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_pmx
    mutation = mutation.mutation_reverse

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            # 上一代种群
            Chrom_old = self.Chrom.copy()
            # 根据种群的基因值（二进制0或1）获得参数值 X （ 种群个体数 * 变量个数n_dim )
            self.X = self.chrom2x(self.Chrom)
            # 获得参数值X 对应的性能 Y
            self.Y = self.x2y()

            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # put parent and offspring together and select the best size_pop number of population
            # 将父亲代和子代放在一起，并选择最佳的人口规模_pop数量
            self.Chrom = np.concatenate([Chrom_old, self.Chrom], axis=0)
            # 根据种群基因获得参数值 X
            self.X = self.chrom2x(self.Chrom)
            # 根据参数值 X 计算性能值 Y
            self.Y = self.x2y()
            # 对种群进行排序
            self.ranking()
            # 将数组x中的元素从小到大排列，按顺序返回对应的索引值
            selected_idx = np.argsort(self.Y)[:self.size_pop]
            # 选择排名靠前的 self.size_pop 个个体作为新的种群
            self.Chrom = self.Chrom[selected_idx, :]

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y.copy())
            self.all_history_FitV.append(self.FitV.copy())

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y
