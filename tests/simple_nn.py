import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.linspace(-1, 1, 200)[:, None]       # [batch, 1]
y = x ** 2                                  # [batch, 1]
learning_rate = 0.001


def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    return 1 - tanh(x)**2


w1 = np.random.uniform(0, 1, (1, 10))
w2 = np.random.uniform(0, 1, (10, 10))
w3 = np.random.uniform(0, 1, (10, 1))

b1 = np.full((1, 10), 0.1)
b2 = np.full((1, 10), 0.1)
b3 = np.full((1, 1), 0.1)


for i in range(300):
    a1 = x
    z2 = a1.dot(w1) + b1
    a2 = tanh(z2)
    z3 = a2.dot(w2) + b2
    a3 = tanh(z3)
    z4 = a3.dot(w3) + b3

    cost = np.sum((z4 - y)**2)/2

    # backpropagation
    z4_delta = z4 - y
    dw3 = a3.T.dot(z4_delta)
    db3 = np.sum(z4_delta, axis=0, keepdims=True)

    z3_delta = z4_delta.dot(w3.T) * derivative_tanh(z3)
    dw2 = a2.T.dot(z3_delta)
    db2 = np.sum(z3_delta, axis=0, keepdims=True)

    z2_delta = z3_delta.dot(w2.T) * derivative_tanh(z2)
    dw1 = x.T.dot(z2_delta)
    db1 = np.sum(z2_delta, axis=0, keepdims=True)

    # update parameters
    for param, gradient in zip([w1, w2, w3, b1, b2, b3], [dw1, dw2, dw3, db1, db2, db3]):
        param -= learning_rate * gradient

    print(cost)

plt.plot(x, z4)
plt.show()