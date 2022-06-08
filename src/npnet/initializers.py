from abc import ABCMeta, abstractmethod

import numpy as np


class BaseInitializer(metaclass=ABCMeta):
    @abstractmethod
    def initialize(self, x):
        pass


class RandomNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)


class RandomUniform(BaseInitializer):
    def __init__(self, low=0., high=1.):
        self._low = low
        self._high = high

    def initialize(self, x):
        x[:] = np.random.uniform(self._low, self._high, size=x.shape)


class Zeros(BaseInitializer):
    def initialize(self, x):
        x[:] = np.zeros_like(x)


class Ones(BaseInitializer):
    def initialize(self, x):
        x[:] = np.ones_like(x)


class TruncatedNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)
        truncated = 2*self._std + self._mean
        x[:] = np.clip(x, -truncated, truncated)


class Constant(BaseInitializer):
    def __init__(self, v):
        self._v = v

    def initialize(self, x):
        x[:] = np.full_like(x, self._v)


random_normal = RandomNormal()
random_uniform = RandomUniform()
zeros = Zeros()
ones = Ones()
truncated_normal = TruncatedNormal()
