import warnings

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)

'''
    新增属性 precisions
    更新时间：2021/1/5  14:15
'''
class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, precisions, random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        '''
            新增属性 precisions
            更新时间：2021/1/5  14:15
        '''
        self._space = TargetSpace(f, pbounds, precisions, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # ----------------------- 新增：性能预测 start -----------------------
        # import pandas as pd
        # import time, os, sys
        # sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
        # print(sys.path)
        # from parameter_choose.main_time_predict import main_time_predict
        #
        #
        # df_param = pd.DataFrame(self._space.params,columns=self._space.keys)
        # df_target = pd.DataFrame(self._space.target ,columns=['runtime'])
        # print('param shape = ' + str(df_param.shape) + ', target shape = ' + str(df_target.shape))
        # df_train = pd.concat([df_param,df_target],axis=1)
        #
        # time_path = str(time.strftime('%Y-%m-%d')) + '/' + str(time.strftime('%H-%M-%S')) + '/'
        # save_path = "./parameter_choose/model/" + time_path
        # train_data_path = './parameter_choose/train_data_csv/'+ time_path
        # train_data_csv = train_data_path + '/train_data.csv'
        # if not os.path.isdir(train_data_path):
        #     os.makedirs(train_data_path)
        #
        # df_train.to_csv(train_data_csv, index=False)
        #
        # predict_time_model = main_time_predict(filepath=train_data_csv, save_path=save_path)
        # predict_time_model.main()
        # ----------------------- 新增：性能预测 end -----------------------
        import numpy as np
        mean_target = np.mean(self._space.target)
        std_target = np.std(self._space.target)


        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        # 忽略高斯过程中发出的警告信息
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 1. 使用目前已经搜索过的参数和target拟合高斯过程回归模型
            self._gp.fit(self._space.params, self._space.target)

        '''
                新增属性 precisions
                更新时间：2021/1/5  14:15
        '''
        # Finding argmax of the acquisition function.
        # 2. 找到采集函数的最大值，赋给 suggestionn
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            precisions=self._space._precisions,
            random_state=self._random_state,
            Tconstraint=np.percentile(self._space.target, 75)
        )
        # 将suggestion封装成配置参数的样式传递回去做probe计算target值并注册到space中
        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        # 实例UtilityFunction，指定acq = ucb
        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)