import numpy as np
import math as m
from scipy.linalg import solve
import matplotlib.pyplot as plt

eps_0 = 0.5
N = 3
M = 3

y = np.array([1.062, 0.792, 0.67], float)
x = np.array([0, 1.5, 3], float)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

A = np.zeros(shape=(M, M))
B = np.zeros(shape=(M))


def element(i, j):
    sum = 0
    for k in range(0, N):
        sum = sum + x[k] ** (i + j)

    return sum


def right_side(i):
    sum = 0
    for k in range(0, N):
        sum = sum + y[k] * (x[k] ** i)
    return sum


for i in range(0, M):
    for j in range(0, M):
        A[i][j] = element(i, j)

for i in range(0, M):
    B[i] = right_side(i)

W = solve(A, B)


def calc_polynomial(x):
    sum = 0
    for i in range(0, M):
        sum = sum + W[i] * x ** i

    return sum


import matplotlib.pyplot as plt

plt.scatter(x, y)

xx = np.arange(0, 10, 0.01)
yy = calc_polynomial(xx)
plt.plot(xx, yy)

plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 1.2)
plt.show()
