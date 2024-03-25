import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def classification(train: np.ndarray, epochs, ed_coef, K, delta, type):
    if type == "step":
        return step_classification(train, epochs, ed_coef)
    elif type == "sigm":
        return sigm_classification(train, epochs, ed_coef, K, delta)


def step_classification(train: np.ndarray, epochs, ed_coef):
    W = np.zeros(len(train[0]))
    train_set = train[:, :2]
    class_data = train[:, 2]
    train_data = np.hstack((np.ones(len(train_set)).reshape(-1, 1), train_set))
    for _ in range(epochs):
        for i in range(len(train_data)):
            res = W @ train_data[i]
            if res <= 0 and class_data[i] == 1:
                W += ed_coef * train_data[i]
            elif res > 0 and class_data[i] == 0:
                W -= ed_coef * train_data[i]

    return W


def sigm_classification(train: np.ndarray, epochs, ed_coef, K, delta):
    train_set = train[:, :2]
    class_data = train[:, 2]
    train_data = np.hstack((np.ones(len(train_set)).reshape(-1, 1), train_set))
    W_min = []
    min_error = math.inf
    for _ in range(epochs):
        W = np.random.rand(len(train[0]))
        grad = np.zeros(len(train[0]))

        for k in range(K):  # цикл градиентного спуска
            error = 0
            for i in range(len(train_data)):
                a = W @ train_data[i]
                sigma_a = sigmoid(a)
                error += 0.5 * ((sigma_a - class_data[i]) ** 2)
                grad += train_data[i] * (sigma_a * (1 - sigma_a)) * (sigma_a - class_data[i])  # вектор-градиент
            if error <= delta:
                W_min = W.copy()
                min_error = error
                break

            W -= ed_coef * grad

            if k == (K - 1) and error < min_error:
                W_min = W.copy()
                min_error = error

    return W_min
