import numpy as np
import neuralnets as nn
import matplotlib.pyplot as plt

np.random.seed(1)
x0 = np.random.normal(-2, 1, (100, 2))
x1 = np.random.normal(2, 1, (100, 2))
y0 = np.zeros((100, 1), dtype=np.int32)
y1 = np.ones((100, 1), dtype=np.int32)
x = np.concatenate((x0, x1), axis=0)
y = np.concatenate((y0, y1), axis=0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        w_init = nn.init.RandomUniform()
        b_init = nn.init.Constant(0.1)

        self.l1 = nn.layers.Dense(2, 10, nn.act.tanh, w_init, b_init)
        self.l2 = nn.layers.Dense(10, 10, nn.act.tanh, w_init, b_init)
        self.out = nn.layers.Dense(10, 1, nn.act.sigmoid)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o


net = Net()
opt = nn.optim.Adam(net.params, lr=0.1)
loss_fn = nn.losses.SigmoidCrossEntropy()

for _ in range(1000):
    o = net.forward(x)
    loss = loss_fn(o, y)
    net.backward(loss)
    opt.step()
    print(loss)

plt.scatter(x[:, 0], x[:, 1], c=o.data.ravel(), s=100, lw=0, cmap='RdYlGn')
plt.show()

