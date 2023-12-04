import math
from typing import Any
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3*x**2 - 4*x+5


xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
# plt.show()

class Value:
    def __init__(self, data, _children=(), _op='', label='') -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label if label else _op

    def __repr__(self) -> str:
        if len(self._prev) > 0:
            # should prepend label here, but gets bloated
            return f'[{self.label}:{self.data}:{round(self.grad, 4)}]{self._op.join([repr(x) for x in self._prev])}'
        else:
            return f'({self.label}:{self.data}:{round(self.grad, 4)})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label='?')
        out = Value(self.data+other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        out = self + other
        out.label= '+'
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label='?')
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        out = self * other
        out.label= '?'
        return out

    def __truediv__(self, other):
        out = self * other**-1
        out.label= '/'
        return out

    def __neg__(self):
        out = self * -1
        out.label= '-'
        return out

    def __sub__(self, other):
        out = self + (-other)
        out.label= '-'
        return out

    def __pow__(self, other):
        x = self.data ** other
        out = Value(x, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other -1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    #  Should break if you feed the same node to two different outputs
    #  We haven't looked at how this works, but it has to because one node can have many outputs
    #  Should the proper application of the chain rule be the sum of the chained derivatives?
    #  Whoops got there.
    # def do_backprop(self):
    #     self.grad = 1
    #     self.backprop()

    # def backprop(self):
    #     self._backward()
    #     for x in self._prev:
    #         x.backprop()

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

a = Value(3.0, label='a')
b = Value(-2.0, label='b')
c = Value(10, label='c')
e = a*b; e.label = 'e'
d = e+c; d.label = 'd'
f = Value(0.15, label='f')
L = d*f; L.label = 'L'
ex = (2*L).exp()
test = (ex - 1) / (ex + 1)
# test = L.tanh();
test.label = 'tanh'
test.backward()
# test.do_backprop()
# test.grad = 1.0
# test._backward()
# L._backward()
# f._backward()
# d._backward()
# e._backward()
# c._backward()
# a._backward()
# b._backward()
print(test)
dbla = Value(1.0, label='a')
newb = 1 + dbla + dbla; newb.label ='b'
newb.backward()
print(newb)


# so this kept annoying me, I need to look at his actual code to see if this is fixed or find out if it's not a bug.
# I think it's debatably a bug, but it's not the purpose of the library so it really probably doesn't matter.
for i in range(10):
    print(Value(1.0) / Value(2.0))

testa = Value(3.0)
testb = Value(4.0)
testc = testa*testb
print(testc)
testa.data = 2.0
print(testc)

# import torch

# x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
# x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
# w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
# w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
# b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
# n = x1*w1 + x2*w2 + b
# o = torch.tanh(n)

# print(o.data.item())
# o.backward()

# print('---')
# print('x2', x2.grad.item())
# print('w2', w2.grad.item())
# print('x1', x1.grad.item())
# print('w1', w1.grad.item())

import random

class Neuron:

    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        act = sum((wi*xi for wi, xi in zip(self.w, *args)),  self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        outs = [n(*args) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

    def __init__(self, nin, nouts) -> None:
        sz = [nin]+nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
# print(n(x).data)

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(100):
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
    if k % 10 == 0:
        print(f"{k}:{loss.data:.4f}")

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    if loss.data < 0.0001:
        break

    for p in n.parameters():
        p.data += -0.1 * p.grad

print("final")
for y in ypred:
    print(y.data)