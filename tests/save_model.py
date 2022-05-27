import numpy as np
import matplotlib.pyplot as plt

import npnet


np.random.seed(1)
x = np.linspace(-1, 1, 200)[:, None]       # [batch, 1]
y = x ** 2 + np.random.normal(0., 0.1, (200, 1))     # [batch, 1]


class Net(npnet.Module):
    def __init__(self):
        super().__init__()
        self.l1 = npnet.layers.Dense(1, 10, npnet.act.tanh)
        self.out = npnet.layers.Dense(10, 1, )

    def forward(self, x):
        x = self.l1(x)
        o = self.out(x)
        return o


net1 = Net()
opt = npnet.optim.Adam(net1.params, lr=0.1)
loss_fn = npnet.losses.MSE()

for _ in range(1000):
    o = net1.forward(x)
    loss = loss_fn(o, y)
    net1.backward(loss)
    opt.step()
    print(loss)

# save net1 and restore to net2
net1.save("./params.pkl")
net2 = Net()
net2.restore("./params.pkl")
o2 = net2.forward(x)

plt.scatter(x, y, s=20)
plt.plot(x, o2.data, c="red", lw=3)
plt.show()

