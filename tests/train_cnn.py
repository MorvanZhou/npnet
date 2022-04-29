import npnet
import numpy as np

np.random.seed(1)
f = np.load('./mnist.npz')
train_x, train_y = f['x_train'][:, :, :, None], f['y_train'][:, None]
test_x, test_y = f['x_test'][:2000][:, :, :, None], f['y_test'][:2000]

train_loader = npnet.DataLoader(train_x, train_y, batch_size=64)


class CNN(npnet.Module):
    def __init__(self):
        super().__init__()
        self.seq_layers = self.sequential(
            npnet.layers.Conv2D(1, 6, (5, 5), (1, 1), "same", channels_last=True),  # => [n,28,28,6]
            npnet.layers.MaxPool2D(2, 2),  # => [n, 14, 14, 6]
            npnet.layers.Conv2D(6, 16, 5, 1, "same", channels_last=True),  # => [n,14,14,16]
            npnet.layers.MaxPool2D(2, 2),  # => [n,7,7,16]
            npnet.layers.Flatten(),  # => [n,7*7*16]
            npnet.layers.Dense(7 * 7 * 16, 10, )
        )

    def forward(self, x):
        o = self.seq_layers.forward(x)
        return o


cnn = CNN()
opt = npnet.optim.Adam(cnn.params, 0.001)
loss_fn = npnet.losses.SparseSoftMaxCrossEntropyWithLogits()


for step in range(300):
    bx, by = train_loader.next_batch()
    by_ = cnn.forward(bx)
    loss = loss_fn(by_, by)
    cnn.backward(loss)
    opt.step()
    if step % 50 == 0:
        ty_ = cnn.forward(test_x)
        acc = npnet.metrics.accuracy(np.argmax(ty_.data, axis=1), test_y)
        print("Step: %i | loss: %.3f | acc: %.2f" % (step, loss.data, acc))

