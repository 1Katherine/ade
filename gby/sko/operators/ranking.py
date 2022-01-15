import numpy as np


def ranking(self):
    # GA select the biggest one, but we want to minimize func, so we put a negative here
    print('ranking')
    self.FitV = -self.Y


def ranking_linear(self):
    '''
    For more details see [Baker1985]_.

    :param self:
    :return:

    .. [Baker1985] Baker J E, "Adaptive selection methods for genetic
    algorithms, 1985.
    '''
    # np.argsort(-self.Y) 对 -Y 按照从小到大排列，返回索引值
    self.FitV = np.argsort(np.argsort(-self.Y))
    print('ranking_linear中self.FitV =  ' + str(self.FitV))
    return self.FitV
