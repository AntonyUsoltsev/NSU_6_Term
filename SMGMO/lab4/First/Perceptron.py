import numpy as np
import math


class Perceptron:
    def __init__(self, feature_size, num_classes):
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.W = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def sigm_classification(self, train_data, epochs, ed_coef, K, delta):
        W_min = []
        min_error = math.inf

        for _ in range(epochs):
            W = np.random.rand(self.num_classes, self.feature_size)  # W has shape (num_classes, feature_size)
            grad = np.zeros((self.num_classes, self.feature_size))

            for k in range(K):
                error = 0
                for i in range(len(train_data)):
                    a = W @ np.hstack((1, train_data[i, :-1]))  # Add bias and consider only features
                    sigma_a = self.sigmoid(a)
                    error += 0.5 * np.sum((sigma_a - train_data[i, -1]) ** 2)
                    grad += np.outer(sigma_a * (1 - sigma_a) * (sigma_a - train_data[i, -1]),
                                     np.hstack((1, train_data[i, :-1])))

                if error <= delta:
                    W_min = W.copy()
                    min_error = error
                    break

                W -= ed_coef * grad

                if k == (K - 1) and error < min_error:
                    W_min = W.copy()
                    min_error = error

        self.W = W_min

    def predict(self, data):
        if data.shape[1] != self.feature_size - 1:
            raise ValueError(f"Data dimension does not match feature size: {data.shape[1]} != {self.feature_size - 1}")

        data_with_bias = np.hstack((np.ones(len(data)).reshape(-1, 1), data))

        predictions = []
        for i in range(len(data)):
            predictions.append(np.argmax(self.softmax(self.W @ data_with_bias[i].reshape(-1, 1))))
        return predictions
