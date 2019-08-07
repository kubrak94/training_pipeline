from functools import partial

import numpy as np


class AlphaScheduler:
    def __init__(self, base):
        assert 0.5 < base <= 1, 'base should be greater than 0.5 and less than or equal to 1'
        self._base = base
        self.alpha = self._base

    def step(self):
        pass


class Constant(AlphaScheduler):
    pass


class Sigmoid(AlphaScheduler):
    def __init__(self, base, multiplier=2, shift=15):
        super().__init__(base)
        self._epoch = 0

        self._scheduler_function = partial(sigmoid, multiplier=multiplier, shift=shift)
        
        self.alpha = self._base * self._scheduler_function(epoch=self._epoch)
        
    def step(self):
        self._epoch += 1
        self.alpha = self._base * self._scheduler_function(epoch=self._epoch)


def sigmoid(multiplier, shift, epoch):
    a = (epoch - 15) / multiplier
    return (np.exp(a) / (np.exp(a) + 1))