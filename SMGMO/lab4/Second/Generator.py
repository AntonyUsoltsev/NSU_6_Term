import numpy as np


def polynome(a, b, c, d, x):
    return a * x ** 3 + b * x ** 2 + c * x + d


def sinus(x):
    return x * np.sin(2 * np.pi * x)


def generate(a, b, c, d, N, noise, func):
    x = np.random.uniform(-1, 1, N)
    eps = np.random.uniform(-noise, noise, N)
    if func == "sin":
        y = sinus(x) + eps
    elif func == "poly":
        y = polynome(a, b, c, d, x) + eps
    else:
        raise ValueError("Invalid generation type")
    return x, y
