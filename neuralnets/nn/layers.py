import numpy as np
import neuralnets as nn


class Layer:
    def __init__(self):
        self.order = None
        self.name = None
        self.w = None
        self.b = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Dense(Layer):
    def __init__(self, n_in, n_out, activation=None, w_initializer=None, b_initializer=None):
        super().__init__()
        self._n_in = n_in
        self._n_out = n_out
        self.w = np.empty((n_in, n_out), dtype=np.float32)
        self.b = np.empty((1, n_out), dtype=np.float32)

        self._wx_b = None
        self._activated = None
        self._x = None
        self.gradients = {"w": np.empty_like(self.w), "b": np.empty_like(self.b)}

        if activation is None:
            self._a = nn.act.Linear()
        elif isinstance(activation, nn.act.Activation):
            self._a = activation
        else:
            raise ValueError

        if w_initializer is None:
            nn.init.RandomUniform().initialize(self.w)
        elif isinstance(w_initializer, nn.init.BaseInitializer):
            w_initializer.initialize(self.w)
        else:
            raise ValueError

        if b_initializer is None:
            nn.init.Constant(0.1).initialize(self.b)
        elif isinstance(b_initializer, nn.init.BaseInitializer):
            b_initializer.initialize(self.b)
        else:
            raise ValueError

    def forward(self, x):
        if isinstance(x, np.ndarray):
            self.order = 0  # use layer input's information to set layer order
            _x = x.astype(np.float32)
        else:
            self.order = x.info["new_layer_order"]
            _x = x.data
        self._x = _x
        self._wx_b = _x.dot(self.w) + self.b
        if self._a:
            self._activated = self._a(self._wx_b)
        else:
            self._activated = self._wx_b
        out = nn.Variable(self._activated)
        out.info["new_layer_order"] = self.order + 1
        return out

    def backward(self, delta):
        delta = delta * self._a.derivative(self._wx_b)
        gw = self._x.T.dot(delta)
        gb = np.sum(delta, axis=0, keepdims=True)
        delta_ = delta.dot(self.w.T)
        return delta_, gw, gb


