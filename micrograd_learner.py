import math
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
        self.label = label

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
        out.label= '?'
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
        out.label= '?'
        return out

    def __neg__(self):
        out = self * -1
        out.label= '?'
        return out

    def __sub__(self, other):
        out = self + (-other)
        out.label= '?'
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
