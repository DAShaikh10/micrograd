"""
A simple neural network module built on top of `micrograd`.

Reference: https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

import random

from .engine import Value


class Module:
    """
    `PyTorch` style organization of deep neural network base class.
    Serves as the base class but currently only is responsible for holding functionality
    to reset gradients before back propagation i.e. `backward()`
    """

    def zero_grad(self):
        """
        Reset gradient of all parameters to `0.`.
        Required before performing back propagation repeatedly.
        """

        for parameter in self.parameters():
            parameter.grad = 0

    def parameters(self):
        """
        Get all the learnable parameters of a neural network.
        `PyTorch` convention is followed.
        """

        return []


class Neuron(Module):
    """
    `PyTorch` stlye simple `Neuron` class that represents the fundamental unit of a MLP.
    """

    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0)

    def __call__(self, x):
        pre_activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return pre_activation.tanh()

    def __repr__(self):
        return f"Linear Neuron({len(self.w)})"

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """
    `PyTorch` style simple `Layer` class to represent a layer of neurons within a MLP.
    """

    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def parameters(self):
        """
        Get all the learnable parameters of a neural network's individual layer.
        `PyTorch` convention is followed.
        """
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]


class MLP:
    """
    `PyTorch` style simple `MLP` class to represent MLP neural network.
    """

    def __init__(self, n_inputs, n_outputs):
        sz = [n_inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def parameters(self):
        """
        Get all the learnable parameters of a MLP neural network.
        `PyTorch` convention is followed.
        """
        return [parameter for layer in self.layers for parameter in layer.parameters()]
