import numpy as np


class Variable:
    def __init__(self, v):
        self.data = v
        self._error = np.empty_like(v)   # for backpropagation of the last layer
        self.info = {}

    def __repr__(self):
        return str(self.data)

    def set_error(self, error):
        assert self._error.shape == error.shape
        self._error[:] = error

    @property
    def error(self):
        return self._error

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim