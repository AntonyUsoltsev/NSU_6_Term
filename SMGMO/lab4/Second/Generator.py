import numpy as np


def polynome(a, b, c, d, x):
    return a * x ** 3 + b * x ** 2 + c * x + d


def sinus(x):
    return x * np.sin(2 * np.pi * x)


def generate(a, b, c, d, N, type):
    y = np.zeros(shape=(N), dtype='f')
    x = np.random.uniform(-1, 1, N)
    if type == "sin":
        for i in range(N):
            eps = 0
            # np.random.uniform(-0.1, 0.1))
            y[i] = sinus(x[i]) + eps
        return x, y
    elif type == "poly":
        for i in range(N):
            eps = 0
            # np.random.uniform(-0.1, 0.1))
            y[i] = polynome(a, b, c, d, x[i]) + eps
        return x, y
