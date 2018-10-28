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
            delta, grads = layer.backward(delta)
            self.params[layer.name]["grads"]["w"][:] = grads["gw"]
            if layer.use_bias:
                self.params[layer.name]["grads"]["b"][:] = grads["gb"]

    def save(self, path):
        saver = nn.Saver()
        saver.save(self, path)

    def restore(self, path):
        saver = nn.Saver()
        saver.restore(self, path)

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, nn.layers.Layer):
            layer = value
            self.params[key] = {
                "vars": {"w": layer.w},
                "grads": {"w": np.empty_like(layer.w)}
            }
            if layer.use_bias:
                self.params[key]["vars"]["b"] = layer.b
                self.params[key]["grads"]["b"] = np.empty_like(layer.b)

        object.__setattr__(self, key, value)
