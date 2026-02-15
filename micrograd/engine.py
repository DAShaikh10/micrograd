"""
A simple `micrograd` module built as an condensed inspiration from PyTorch's autograd.

This module is built by following Andrej Karpathy's explanation:
https://www.youtube.com/watch?v=VMj-3S1tku0&pp=ygUPYW5kcmVqIGthcnBhdGh5

Reference: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
"""

import math

# pylint: disable=protected-access


class Value:
    """
    Stores a single scalar value and its gradient
    """

    def __init__(self, data: float, _children: set = (), op=None):
        # Internal variables used for autograd graph construction.
        self._backward = lambda: None
        self._op = op  # The op that produced this node, for graphviz / debugging / etc.
        self._prev = set(_children)

        self.data = data
        self.grad = 0

    def __add__(self, other):  # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            """
            if `f(a, b) = a + b`
            then, `df/da = 1` and `df/db = 1`
            """
            self.grad += out.grad  # self.grad += 1.0 * out.grad
            other.grad += out.grad  # other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            """
            if `f(a, b) = a * b`
            then, `df/da = b` and `df/db = a`
            """
            self.grad += other.data * out.grad  # self.grad += other.data * out.grad
            other.grad += self.data * out.grad  # other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        # Python cannot perform 2 * Value(1.0) i.e. `int.__(Value)`,
        # hence we add the __rmul__ method to support this operation.
        # This is called reverse multiplication and is used when the left operand
        # does not support the multiplication operation with the right operand.
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __pow__(self, other):  # self ** other
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), "**")

        def _backward():
            """
            if `f(a, b) = a ** b`
            then, `df/da = b * a ** (b - 1)`
            """

            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __truediv__(self, other):  # self / other
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def backward(self):
        """
        Perform backpropagation with respect to `self` in a Directed Acyclic Graph (DAG)
        """

        # Topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # If `f = a`, then df/da = 1
        self.grad = 1.0

        # Go one variable at a time and apply the chain rule to get its gradient.
        for node in reversed(topo):
            node._backward()

    def exp(self):
        """
        Implementation of exponentiation operation for `Value` object(s).
        """

        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            """
            if `f(a) = exp(a)`
            then `df/da = exp(a)`
            """
            self.grad += out.data * out.grad  # self.grad += exp(self.data) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        """
        Implementation of `tanh` for `Value` object.

        `tanh(x) = (e ** (2x) - 1) / (e ** (2x) + 1)`
        """

        out = Value(
            (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1),
            (self,),
            "tanh",
        )

        def _backward():
            """
            if `f(a) = tanh(a)`
            then, df/da = 1 - tanh(a) ** 2
            """
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out
