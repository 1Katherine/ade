import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

'''
        新增属性 precisions
        更新时间：2021/1/5  14:15
'''
# 预测的执行时间
predict_runtime = []
# ei从大到小顺序下标
ei_argsort_list = []
def acq_max(ac, gp, y_max, bounds, precisions, Tconstraint, random_state, n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """
    # Warm up with random points
    x_tries = np.empty((n_warmup, bounds.shape[0]))
    '''
            新增属性 precisions
            方法：根据精度生成n_warmup（10000个）随机样本点，赋给x_tries
            在acqusition function上随机生成对应精度的样本点，用于预测下一个最好样本（主要有两种精度 1.0 和 0.01 ）
            更新时间：2021/1/5  14:15
    '''
    bounds_and_pre = np.column_stack((bounds, precisions))
    # print('bounds = ' + str(bounds))
    for i in range(n_warmup):
        for col, (lower, upper, pre) in enumerate(bounds_and_pre):
            if pre == 1.0:
                x_tries[i][col] = np.round(random_state.uniform(lower, upper, size=1))
            if pre == 0.01:
                x_tries[i][col] = np.round(random_state.uniform(lower, upper, size=1),2)
        # print('x_tries[i] = ' + str(x_tries[i]))


    '''
            注释源代码
            注释时间：2021/1/5  14:15
    '''
    # x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
    #                                size=(n_warmup, bounds.shape[0]))

    # 使用训练好的gp模型和采集函数预测x_tries对应的y值 ys.shape = (10000,)
    # ac=utility_function.utility , 给定x样本点，返回gp预测的y
    ys = ac(x_tries, gp=gp, y_max=y_max)
    global ei_argsort_list
    ei_argsort_list = np.argsort(ys)

    print('将ei值按从大到小的顺序排列，返回从大到小的索引值，【0】为ei值最大的索引下标，以此类推 . ei_argsort_list = ' + str(ei_argsort_list))
    print('对x_tries的10000个随机样本进行gp.predict . predict_runtime = ' + str(predict_runtime))

    print('使用已有样本空间中的target计算约束的执行时间，可以是均值，25%分位数，75%分位数 Tconstraint = ' + str(Tconstraint))

    idx = 0
    while idx < x_tries.shape[0] - 1 and -predict_runtime[ei_argsort_list[idx]] > -Tconstraint:
        # print('idx = ' + str(idx))
        idx += 1
    if idx == x_tries.shape[0] - 1:
        print('没在x_tries中找到预测的执行时间少于Tcontraint的随机样本，按照原有策略选出x_max')
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
    else:
        print('在x_tries中找到预测的执行时间少于Tcontraint的随机样本并且该样本的ei值最大，按照新策略选出x_max')
        x_max = x_tries[ei_argsort_list[idx]]
        max_acq = ys[ei_argsort_list[idx]]
    print('idx = ' + str(idx) + ' ,x_max 对应ei中的idx下标 = ' + str(ei_argsort_list[idx]))
    # print('ac_ys = ' + str(ys))
    # print('ys.argmax() = ' + str(ys.argmax()))
    # # argmax最大值所对应的索引， 取出预测的最大y值对应的样本x
    # x_max = x_tries[ys.argmax()]
    # # 记录采集函数找到的最大样本点，用于explore样本点时的更新使用
    # max_acq = ys.max()
    # print('x_max1 = ' + str(x_max))
    print('x_max = ' + str(x_max) + ',max_acq = ' + str(max_acq) + ',predict_runtime = ' + str(predict_runtime[ei_argsort_list[idx]]))


    '''
                注释源代码
                注释时间：2021/1/5  14:15
    '''
    # Explore the parameter space more throughly
    # x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
    #                                size=(n_iter, bounds.shape[0]))
    # explore 随机产生n_iter个（10个）样本点，看看能不能找到更好的样本点赋给x_max
    x_seeds = np.empty((n_iter, bounds.shape[0]))
    '''
            新增属性 precisions
            方法：根据精度随机生成对应精度的数据，用于更全面地探索参数空间（10个）（主要有两种精度 1.0 和 0.01 ）
            更新时间：2021/1/5  14:15
    '''
    for i in range(n_iter):
        for col, (lower, upper, pre) in enumerate(bounds_and_pre):
            # print('pre = ' + str(pre))
            if pre == 1.0:
                x_seeds[i][col] = np.round(random_state.uniform(lower, upper, size=1))
            if pre == 0.01:
                x_seeds[i][col] = np.round(random_state.uniform(lower, upper, size=1), 2)
        # print('x_seeds[i] = ' + str(x_seeds[i]))

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function 求-ac的极小值 = 求ac的极大值
        # minimize ： 非线性规划（求极小值） , x = x_try.reshape(1, -1)
        # 求 函数 -ac(x.reshape(1, -1), gp=gp, y_max=y_max) 的最小值对应的样本res.x
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")
        # See if success
        if not res.success:
            continue
        # Store it if better than previous minimum(maximum).
        # 如果找到了比max_acq还要大的样本x，则更新max_acq
        if max_acq is None or -res.fun[0] >= max_acq:
            # print('res.x = ' + str(res.x))
            for row, (num, pre) in enumerate(np.column_stack((res.x, precisions))):
                if pre == 1.0:
                    # 四舍五入取整数
                    x_max[row] = round(num)
                if pre == 0.01 :
                    x_max[row] = round(num, 2)
            # x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    # np.clip将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min\
    print('找到的下一个探索样本点的预测执行时间为' + str(gp.predict([x_max])))
    return x_max
    # return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """
    # kind : 采集函数的类型（ucb、ei、poi）
    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi
        
        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std

        global predict_runtime
        predict_runtime_temp = gp.predict(x)
        predict_runtime = predict_runtime_temp.tolist()

        return a * norm.cdf(z) + std * norm.pdf(z)


    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)

# --------------- 新增预测执行时间的代码 start---------------
# def get_predict_runtime(x, model_path):
#     from sklearn.ensemble import RandomForestRegressor
#     import joblib
#     name = 'rf'
#     runtime = 0
#     warnings.filterwarnings("ignore")
#     if name.lower() == "rf":
#         model = joblib.load(model_path + '/rf/time_predict_rf.pkl')
#         # print('x = ' + str(x))
#         runtime = model.predict(np.matrix(x))[0]
#         # print('runtime = ' + str(runtime))
#     return runtime
# --------------- 新增预测执行时间的代码 end---------------

def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)
