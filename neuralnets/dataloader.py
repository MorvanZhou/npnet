import numpy as np


class DataLoader:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.bs = batch_size
        self.p = 0
        self.bg = self.batch_generator()

    def batch_generator(self):
        while True:
            p_ = self.p + self.bs
            if p_ > len(self.x):
                self.p = 0
                continue
            if self.p == 0:
                indices = np.random.permutation(len(self.x))
                self.x[:] = self.x[indices]
                self.y[:] = self.y[indices]
            bx = self.x[self.p:p_]
            by = self.y[self.p:p_]
            self.p = p_
            yield bx, by

    def next_batch(self):
        return next(self.bg)
