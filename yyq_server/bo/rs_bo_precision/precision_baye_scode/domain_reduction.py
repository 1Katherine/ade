import numpy as np
from .target_space import TargetSpace


class DomainTransformer():
    '''The base transformer class'''

    def __init__(self, **kwargs):
        pass

    def initialize(self, target_space: TargetSpace):
        # 如果这个方法没有被子类重写，但是调用了，就会报错
        raise NotImplementedError

    def transform(self, target_space: TargetSpace):
        raise NotImplementedError


class SequentialDomainReductionTransformer(DomainTransformer):
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation‐based optimization"
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9
    ) -> None:
        # 振荡的收缩参数 [0.5-0.7]，默认值为 0.7
        self.gamma_osc = gamma_osc
        # pan 的平移参数 通常为1.0，缺省值为1.0
        self.gamma_pan = gamma_pan
        # 缩放参数，默认值 = 0.9
        self.eta = eta
        pass

    def initialize(self, target_space: TargetSpace) -> None:
        """Initialize all of the parameters"""

        # 获取参数的原边界。深拷贝,拷贝前的地址和拷贝后的地址不一样，改变值后不会影响原来的变量
        self.original_bounds = np.copy(target_space.bounds)
        # 指定将来需要缩减的边界
        self.bounds = [self.original_bounds]

        # 按列取平均值
        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        # 上界 - 下界
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]

        # ‘\’表示换行符。通常当一行无法容纳一句代码时，这么表示
        self.previous_d = 2.0 * \
            (self.current_optimal - self.previous_optimal) / self.r

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        # 收缩比例
        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _update(self, target_space: TargetSpace) -> None:

        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        # argmax 返回一个numpy数组中最大值的索引值
        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: np.array, global_bounds: np.array) -> np.array:
        # 按行读取new_bounds，每一行代表一个参数值的范围上下界
        # variable为 1*2 矩阵
        for i, variable in enumerate(new_bounds):
            '''
                新增 下界 > 上界，交换上下界
            '''
            if variable[0] > variable[1]:
                temp = variable[1]
                variable[1] = variable[0]
                variable[0] = temp
            # global_bounds是原始的参数上下界
            # 如果这个参数值的新下界比原始的下界还小（防止新范围越界初始范围）
            if variable[0] < global_bounds[i, 0]:
                variable[0] = global_bounds[i, 0]
            # 如果这个参数值的新上界比原始的上界还大（防止新范围越界初始范围）
            if variable[1] > global_bounds[i, 1]:
                variable[1] = global_bounds[i, 1]
            # 针对x_tries[i][col] = np.random.randint(lower, upper ,size=1)，lower和upper相差不能少于1


        return new_bounds

    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        # 枚举所有参数，记录bounds
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    '''
        开始转换
    '''
    def transform(self, target_space: TargetSpace) -> dict:
        # 更新 收缩比例
        self._update(target_space)

        # 通过r计算出新的边界值，下界为current_optimal - 0.5 * self.r； 上界为 current_optimal + 0.5 * self.r
        new_bounds = np.array(
            [
                self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r
            ]
        ).T

        # 开始转换
        self._trim(new_bounds, self.original_bounds)
        # 将 new_bounds 的下界设为最小值，上界设为最大值
        self.bounds.append(new_bounds)
        # 更改参数的bounds为 new_bounds
        return self._create_bounds(target_space.keys, new_bounds)
