"""
Micrograd is a simplistic implementation of PyTorch's autograd module.
Its goal is to look at the back propagation in a simple setup to easily understand it.
Developed by Andrej Karpathy.
"""

from .engine import Value

__all__ = ["Value"]
