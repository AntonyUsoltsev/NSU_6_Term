import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def classification(train: np.ndarray, epochs, ed_coef, K, delta, type, weigth_size):
    if type == "step":
        return step_classification(train, epochs, ed_coef, weigth_size)
    elif type == "sigm":
        return sigm_classification(train, epochs, ed_coef, K, delta, weigth_size)
    else:
        raise ValueError("Invalid classification type")


def prepare_data(train: np.ndarray, feature_size):
    train_set = train[:, :2]
    class_data = train[:, 2]
    if feature_size == 3:  # Получим линию
        train_data = np.hstack((np.ones(len(train_set)).reshape(-1, 1), train_set))
    elif feature_size == 5:  # Получим кривую
        train_data = np.hstack((np.ones(len(train_set)).reshape(-1, 1), train_set, train_set ** 2))
    elif feature_size == 6:  # Получим кривую
        train_data = np.hstack((np.ones(len(train_set)).reshape(-1, 1), train_set, train_set ** 2,
                                (train_set[:, 0] * train_set[:, 1]).reshape(-1, 1)))
    else:
        raise ValueError("Invalid feature size")
    return train_data, class_data


def step_classification(train: np.ndarray, epochs, ed_coef, feature_size):
    train_data, class_data = prepare_data(train, feature_size)
    W = np.zeros(len(train_data[0]))
    for _ in range(epochs):
        for i in range(len(train_data)):
            res = W @ train_data[i]
            if res <= 0 and class_data[i] == 1:
                W += ed_coef * train_data[i]
            elif res > 0 and class_data[i] == 0:
                W -= ed_coef * train_data[i]

    return W


def sigm_classification(train: np.ndarray, epochs, ed_coef, K, delta, feature_size):
    train_data, class_data = prepare_data(train, feature_size)
    W_min = []
    min_error = math.inf
    for _ in range(epochs):
        W = np.random.rand(len(train_data[0]))
        grad = np.zeros(len(train_data[0]))

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
