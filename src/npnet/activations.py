import numpy as np


class Activation:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, *inputs) -> np.ndarray:
        return self.forward(*inputs)


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), np.full_like(x, self.alpha))


class ELU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(x, self.alpha*(np.exp(x)-1))

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), self.forward(x) + self.alpha)


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(np.tanh(x))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1./(1.+np.exp(-x))

    def derivative(self, x):
        f = self.forward(x)
        return f*(1.-f)


class SoftPlus(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.log(1. + np.exp(x))

    def derivative(self, x):
        return 1. / (1. + np.exp(-x))


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x, axis=-1):
        shift_x = x - np.max(x, axis=axis, keepdims=True)   # stable softmax
        exp = np.exp(shift_x + 1e-6)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def derivative(self, x):
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)


relu = ReLU()
leakyrelu = LeakyReLU()
elu = ELU()
tanh = Tanh()
sigmoid = Sigmoid()
softplus = SoftPlus()
softmax = SoftMax()

