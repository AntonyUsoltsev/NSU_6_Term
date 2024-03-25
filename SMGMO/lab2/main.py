import time
from sklearn.metrics import confusion_matrix
import Generator as generator
import numpy as np
import matplotlib.pyplot as plt
import Perceptron as perceptron


def draw_set(set_to_draw):
    class_0 = set_to_draw[set_to_draw[:, 2] == 0]
    class_1 = set_to_draw[set_to_draw[:, 2] == 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', alpha=0.5)
    plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 1', alpha=0.5)


def draw_separate_line(coef_matr):
    k, a, b = coef_matr[0], coef_matr[1], coef_matr[2]
    if b != 0:
        x_values = np.linspace(-5, 5, 2)
        y_values = (-k - a * x_values) / b
    else:
        y_values = np.linspace(-5, 5, 2)
        x_values = -k / a
    plt.plot(x_values, y_values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.legend()
    plt.ylim(-3, 3)


def class_split(data, coefs):
    train_set = data[:, :2]
    train_data = np.hstack((np.ones(len(train_set)).reshape(-1, 1), train_set))
    return np.hstack((train_set, (train_data @ coefs > 0.0).astype(int).reshape(-1, 1)))


def main():
    num_points = 100
    noise = 0
    epochs = 1000
    ed_coef = 0.5

    K = 10  # K шагов градиентного спуска
    delta = 0.05  # Порог точности локального минимума

    # INFO: train_set = (x,y,c) c -- класс: 1 или 0
    train_set = generator.generate(num_points, noise, "gauss")

    draw_set(train_set)
    plt.show()

    start_time = time.time()
    W = perceptron.classification(train_set, epochs, ed_coef, K, delta, "sigm")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    set_after_class = class_split(train_set, W)
    # INFO: (i, j) - i-ый класс распознан как j
    cm = confusion_matrix(train_set[:, 2], set_after_class[:, 2])
    print(cm)

    draw_set(set_after_class)
    draw_separate_line(W)
    plt.show()


main()
