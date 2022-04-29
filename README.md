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
import npnet

class Net(npnet.Module):
    def __init__(self):
        super().__init__()
        self.l1 = npnet.layers.Dense(n_in=1, n_out=10, activation=npnet.act.tanh)
        self.out = npnet.layers.Dense(10, 1)

    def forward(self, x):
        x = self.l1(x)
        o = self.out(x)
        return o
```

The training procedure starts by defining an optimizer and loss.

```python
net = Net()
opt = npnet.optim.Adam(net.params, lr=0.1)
loss_fn = npnet.losses.MSE()

for _ in range(1000):
    o = net.forward(x)
    loss = loss_fn(o, y)
    net.backward(loss)
    opt.step()
```



## Demo
* A naked and step-by-step [network](https://github.com/MorvanZhou/npnet/tree/master/tests/simple_nn.py) without using my module.
* [Train regressor](https://github.com/MorvanZhou/npnet/tree/master/tests/train_regressor.py)
* [Train classifier](https://github.com/MorvanZhou/npnet/tree/master/tests/train_classifier.py)
* [Train CNN](https://github.com/MorvanZhou/npnet/tree/master/tests/train_cnn.py)
* [Save and restore a trained net](https://github.com/MorvanZhou/npnet/tree/master/tests/save_model.py)


## Install

```
pip install npnet
```

## Download or fork
Download [link](https://github.com/MorvanZhou/npnet/archive/master.zip)

Fork this repo:
```
$ git clone https://github.com/MorvanZhou/npnet.git
```

## Results
![img](https://raw.githubusercontent.com/MorvanZhou/npnet/master/demo.png)
