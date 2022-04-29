import numpy as np
import npnet
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.linspace(-1, 1, 200)[:, None]       # [batch, 1]
y = x ** 2 + np.random.normal(0., 0.1, (200, 1))     # [batch, 1]


class Net(npnet.Module):
    def __init__(self):
        super().__init__()
        w_init = npnet.init.RandomUniform()
        b_init = npnet.init.Constant(0.1)

        self.l1 = npnet.layers.Dense(1, 10, npnet.act.tanh, w_init, b_init)
        self.l2 = npnet.layers.Dense(10, 10, npnet.act.tanh, w_init, b_init)
        self.out = npnet.layers.Dense(10, 1, w_initializer=w_init, b_initializer=b_init)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o


net = Net()
opt = npnet.optim.Adam(net.params, lr=0.1)
loss_fn = npnet.losses.MSE()

for step in range(100):
    o = net.forward(x)
    loss = loss_fn(o, y)
    net.backward(loss)
    opt.step()
    print("Step: %i | loss: %.5f" % (step, loss.data))

plt.scatter(x, y, s=20)
plt.plot(x, o.data, c="red", lw=3)
plt.show()

