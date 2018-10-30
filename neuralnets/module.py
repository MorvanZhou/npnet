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
            if not isinstance(v, nn.layers.BaseLayer):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        dz = loss.delta
        for layer in self._ordered_layers[::-1]:
            dz, grads = layer.backward(dz)
            if grads:
                self.params[layer.name]["grads"]["w"][:] = grads["gw"]
                if layer.__dict__.get("use_bias"):
                    self.params[layer.name]["grads"]["b"][:] = grads["gb"]

    def save(self, path):
        saver = nn.Saver()
        saver.save(self, path)

    def restore(self, path):
        saver = nn.Saver()
        saver.restore(self, path)

    def sequential(self, *layers):
        assert isinstance(layers, (list, tuple))
        for i, l in enumerate(layers):
            self.__setattr__("layer_%i" % i, l)
        return SeqLayers(layers)

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, nn.layers.ParamLayer):
            layer = value
            self.params[key] = {
                "vars": {"w": layer.w},
                "grads": {"w": np.empty_like(layer.w)}
            }
            if layer.use_bias:
                self.params[key]["vars"]["b"] = layer.b
                self.params[key]["grads"]["b"] = np.empty_like(layer.b)

        object.__setattr__(self, key, value)


class SeqLayers:
    def __init__(self, layers):
        assert isinstance(layers, (list, tuple))
        for l in layers:
            assert isinstance(l, nn.layers.BaseLayer)
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)