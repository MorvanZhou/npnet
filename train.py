import numpy as np
import NeuralNets as nn

x = np.linspace(-1, 1, 200)[:, None]       # [batch, 1]
y = x ** 2                                  # [batch, 1]
learning_rate = 0.001

w_init = nn.init.RandomUniform()
b_init = nn.init.RandomUniform()

l1 = nn.layers.Dense(x, 100, nn.act.relu, w_init, b_init)
l2 = nn.layers.Dense(l1, 100, nn.act.relu, w_init, b_init)
out = nn.layers.Dense(l2, 1, w_initializer=w_init, b_initializer=b_init)


cost = out - y




