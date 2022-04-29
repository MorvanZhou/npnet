import numpy as np


class Activation:
    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Activation):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), np.full_like(x, self.alpha))


class ELU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(x, self.alpha*(np.exp(x)-1))

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), self.forward(x) + self.alpha)


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(np.tanh(x))


class Sigmoid(Activation):
    def forward(self, x):
        return 1./(1.+np.exp(-x))

    def derivative(self, x):
        f = self.forward(x)
        return f*(1.-f)


class SoftPlus(Activation):
    def forward(self, x):
        return np.log(1. + np.exp(x))

    def derivative(self, x):
        return 1. / (1. + np.exp(-x))


class SoftMax(Activation):
    def forward(self, x, axis=-1):
        shift_x = x - np.max(x, axis=axis, keepdims=True)   # stable softmax
        exp = np.exp(shift_x + 1e-6)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def derivative(self, x):
        return np.ones_like(x)


relu = ReLU()
leakyrelu = LeakyReLU()
elu = ELU()
tanh = Tanh()
sigmoid = Sigmoid()
softplus = SoftPlus()
softmax = SoftMax()

