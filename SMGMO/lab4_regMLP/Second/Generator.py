import numpy as np


class DataGenerator:
    def __init__(self, generation_type, coefs=None):
        self.generation_type = generation_type
        self.coefs = coefs

    def polynome(self, x):
        degree = len(self.coefs) - 1
        result = sum(coeff * x ** (degree - i) for i, coeff in enumerate(self.coefs))
        return result

    def sinus(self, x):
        return x * np.sin(2 * np.pi * x)

    def generate(self, N, noise):
        x = np.random.uniform(-1, 1, N)
        eps = np.random.uniform(-noise, noise, N)
        if self.generation_type == "sin":
            y = self.sinus(x) + eps
        elif self.generation_type == "poly":
            y = self.polynome(x) + eps
        else:
            raise ValueError("Invalid generation type")
        return x, y

    def generate_function(self):
        x = np.arange(-1, 1, 0.01)
        if self.generation_type == "sin":
            y = self.sinus(x)
        elif self.generation_type == "poly":
            y = self.polynome(x)
        else:
            raise ValueError("Invalid function generation type")
        return x, y
