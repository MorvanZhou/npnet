import numpy as np
import NeuralNets as nn


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Dense(Layer):
    def __init__(self, input_, units, activation, w_initializer=None, b_initializer=None):
        self._x = input_
        self._units = units
        self._a = activation
        self._w = np.empty((input_.shape[1], units), dtype=np.float32)
        self._b = np.empty((1, units), dtype=np.float32)

        if isinstance(w_initializer, nn.init.BaseInitializer):
            w_initializer.initialize(self._w)
        else:
            nn.init.RandomUniform().initialize(self._w)
        if isinstance(b_initializer, nn.init.BaseInitializer):
            b_initializer.initialize(self._b)
        else:
            nn.init.Constant(0.1).initialize(self._b)

    def forward(self, x):
        out = x.dot(self._w) + self._b
        return out

    def backward(self, x):
        pass

