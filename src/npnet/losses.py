from abc import ABCMeta, abstractmethod

import numpy as np
from npnet.activations import softmax


class Loss:
    def __init__(self, loss, delta):
        self.data = loss
        self.delta = delta

    def __repr__(self):
        return str(self.data)


class LossFunction(metaclass=ABCMeta):
    def __init__(self):
        self._pred = None
        self._target = None

    @abstractmethod
    def apply(self, prediction, target):
        pass

    @property
    @abstractmethod
    def delta(self):
        pass

    def _store_pred_target(self, prediction, target):
        p = prediction.data
        p = p if p.dtype is np.float32 else p.astype(np.float32)
        self._pred = p
        self._target = target

    def __call__(self, prediction, target):
        return self.apply(prediction, target)


class MSE(LossFunction):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        loss = np.mean(np.square(self._pred - t))/2
        return Loss(loss, self.delta)

    @property
    def delta(self):
        t = self._target if self._target.dtype is np.float32 else self._target.astype(np.float32)
        return self._pred - t


class CrossEntropy(LossFunction, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._eps = 1e-6

    @abstractmethod
    def apply(self, prediction, target):
        pass


class SoftMaxCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        loss = - np.mean(np.sum(t * np.log(self._pred), axis=-1))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        # according to: https://deepnotes.io/softmax-crossentropy
        onehot_mask = self._target.astype(np.bool)
        grad = self._pred.copy()
        grad[onehot_mask] -= 1.
        return grad / len(grad)


class SoftMaxCrossEntropyWithLogits(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        sm = softmax(self._pred)
        loss = - np.mean(np.sum(t * np.log(sm), axis=-1))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        grad = softmax(self._pred)
        onehot_mask = self._target.astype(np.bool)
        grad[onehot_mask] -= 1.
        return grad / len(grad)


class SparseSoftMaxCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        target = target.astype(np.int32) if target.dtype is not np.int32 else target
        self._store_pred_target(prediction, target)
        sm = self._pred
        log_likelihood = np.log(sm[np.arange(sm.shape[0]), target.ravel()] + self._eps)
        loss = - np.mean(log_likelihood)
        return Loss(loss, self.delta)

    @property
    def delta(self):
        grad = self._pred.copy()
        grad[np.arange(grad.shape[0]), self._target.ravel()] -= 1.
        return grad / len(grad)


class SparseSoftMaxCrossEntropyWithLogits(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        target = target.astype(np.int32) if target.dtype is not np.int32 else target
        self._store_pred_target(prediction, target)
        sm = softmax(self._pred)
        log_likelihood = np.log(sm[np.arange(sm.shape[0]), target.ravel()] + self._eps)
        loss = - np.mean(log_likelihood)
        return Loss(loss, self.delta)

    @property
    def delta(self):
        grad = softmax(self._pred)
        grad[np.arange(grad.shape[0]), self._target.ravel()] -= 1.
        return grad / len(grad)


class SigmoidCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        p = self._pred
        loss = - np.mean(
            t * np.log(p + self._eps) + (1. - t) * np.log(1 - p + self._eps),
        )
        return Loss(loss, self.delta)

    @property
    def delta(self):
        t = self._target if self._target.dtype is np.float32 else self._target.astype(np.float32)
        return self._pred - t
