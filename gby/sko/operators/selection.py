import numpy as np
def selection_tournament(self, tourn_size=3):
    '''
    Select the best individual among *tournsize* randomly chosen
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    print('selection_tournament')
    FitV = self.FitV
    sel_index = []
    # i = 0 - self.size_pop-1
    for i in range(self.size_pop):
        # aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        # 随机产生 tourn_size 个  [0,size_pop) 整数
        aspirants_index = np.random.randint(self.size_pop, size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom

# 默认在随机选择的*tournsize*中选择最佳个体，与`selection_tournament`相同，但使用numpy的速度更快
def selection_tournament_faster(self, tourn_size=3):
    '''
    Select the best individual among *tournsize* randomly chosen
    Same with `selection_tournament` but much faster using numpy
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    print('selection_tournament_faster')
    aspirants_idx = np.random.randint(self.size_pop, size=(self.size_pop, tourn_size))
    aspirants_values = self.FitV[aspirants_idx]
    winner = aspirants_values.argmax(axis=1)  # winner index in every team
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
    self.Chrom = self.Chrom[sel_index, :]
    print('selection_tournament ,self.Chrom = ' + str(self.Chrom))
    return self.Chrom


def selection_roulette_1(self):
    '''
    Select the next generation using roulette
    :param self:
    :return:
    '''
    print('selection_roulette_1')
    FitV = self.FitV
    FitV = FitV - FitV.min() + 1e-10
    # the worst one should still has a chance to be selected
    sel_prob = FitV / FitV.sum()
    sel_index = np.random.choice(range(self.size_pop), size=self.size_pop, p=sel_prob)
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom


def selection_roulette_2(self):
    '''
    Select the next generation using roulette
    :param self:
    :return:
    '''
    print('selection_roulette_2')
    FitV = self.FitV
    FitV = (FitV - FitV.min()) / (FitV.max() - FitV.min() + 1e-10) + 0.2
    # the worst one should still has a chance to be selected
    sel_prob = FitV / FitV.sum()
    sel_index = np.random.choice(range(self.size_pop), size=self.size_pop, p=sel_prob)
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom

