import numpy as np
import neuralnets as nn


class Module(object):
    def __init__(self):
        self._ordered_layers = []
        self.params = {}

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss):
        assert isinstance(loss, nn.losses.Loss)
        # find net order
        layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, nn.layers.Layer):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        delta = loss.delta
        for layer in self._ordered_layers[::-1]:
            delta, gw, gb = layer.backward(delta)
            self.params[layer.name]["grads"]["w"][:] = gw
            self.params[layer.name]["grads"]["b"][:] = gb

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, nn.layers.Layer):
            layer = value
            self.params[key] = {
                    "vars": {"w": layer.w, "b": layer.b},
                    "grads": {"w": np.empty_like(layer.w), "b": np.empty_like(layer.b)}
                }
        object.__setattr__(self, key, value)
