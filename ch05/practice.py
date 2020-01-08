"""
ex04: sigmoid
"""

import numpy as np


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = 1/ (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout):
        return dout * self.y * (1 - self.y)



