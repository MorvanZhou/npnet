# Simple Neural Networks
This is a repo for building a simple Neural Net based only on **[Numpy](http://www.numpy.org/)**.

The usage is similar to [Pytorch](https://pytorch.org/).
There are only limited codes involved to be functional.
Unlike those popular but complex packages such as Tensorflow and Pytorch,
you can dig into my source codes smoothly.

The main purpose of this repo is for you
to understand the code rather than implementation.
So please feel free to read the codes.


## Simple usage
Build a network with a python class and train it.
```python
import neuralnets as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.layers.Dense(n_in=1, n_out=10, activation=nn.act.tanh)
        self.out = nn.layers.Dense(10, 1)

    def forward(self, x):
        x = self.l1(x)
        o = self.out(x)
        return o
```

The training procedure starts by defining a optimizer and loss.

```python
net = Net()
opt = nn.optim.Adam(net.params, lr=0.1)
loss_fn = nn.losses.MSE()

for _ in range(1000):
    o = net.forward(x)
    loss = loss_fn(o, y)
    net.backward(loss)
    opt.step()
```



## Demo
* A naked and step-by-step [network](/simple_nn.py) without using my module.
* [Train regressor](/train_regressor.py)
* [Train classifier](/train_classifier.py)
* [Train CNN](/train_cnn.py)
* [Save and restore a trained net](/save_model.py)


## Download or fork
Download [link](https://github.com/MorvanZhou/simple-neural-networks/archive/master.zip)

Fork this repo:
```
$ git clone https://github.com/MorvanZhou/simple-neural-networks.git
```

## Results
![img](/demo.png)