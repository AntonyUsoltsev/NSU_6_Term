import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import MLP
import Generator as generator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def plot_classification_result(x_data, predictions, activation):
    class_0_x = x_data[predictions.squeeze() == 0]
    class_1_x = x_data[predictions.squeeze() == 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(class_0_x[:, 0], class_0_x[:, 1], color='red', label='Class 0')
    plt.scatter(class_1_x[:, 0], class_1_x[:, 1], color='blue', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f"{activation.__name__}")
    plt.legend()
    plt.show()


def draw_func(x, y, x2, y2):
    plt.scatter(x, y)
    plt.scatter(x2, y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# Функция для обучения модели
def train_model(model, x_train, y_train, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    error_func = nn.MSELoss()
    for epoch in range(epochs):
        model.train()  # set model to training mode
        optimizer.zero_grad()  # zero the gradients
        outputs = model(x_train)  # forward pass
        loss = error_func(outputs, y_train)  # calculate the loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
    return model  # return the trained model


def cross_validation(input_size, hidden_sizes, output_size, activation, x_data, y_data, k=5, epochs=1000, lr=0.3):
    kf = KFold(n_splits=k)
    accuracies = []
    max_accuracy = 0
    best_train = None
    best_predictions = None

    for train_index, test_index in kf.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        model = MLP(input_size, hidden_sizes, output_size, activation)
        model_predictions = train_model(model, x_train_tensor, y_train_tensor, epochs, lr)
        draw_func(x_train, model_predictions.detach().numpy(), x_train, y_train)

        print("Cross validation iteration end\n")

    #     with torch.no_grad():
    #         outputs = model(x_test_tensor)
    #         _, predictions = torch.max(outputs, 1)
    #         accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
    #         accuracies.append(accuracy)
    #
    #         if accuracy > max_accuracy:
    #             max_accuracy = accuracy
    #             best_train = x_train_tensor
    #             _, best_predictions = torch.max(model_predictions, 1)
    #
    # avg_accuracy = np.mean(accuracies)
    # print(f'Average accuracy across {k}-fold cross-validation: {avg_accuracy}')
    # print(f'Max accuracy across {k}-fold cross-validation: {max_accuracy}\n')

    return best_train, best_predictions


def main():
    num_points = 400
    noise = 0.0
    epochs = 2000
    learning_rate = 0.003
    cross_validation_count = 5

    # Задание данных для обучения
    x_train, y_train = generator.generate(3, -2, 3, -2, num_points, "poly")
    # draw_func(x_train, y_train)
    # Задание параметров модели и обучение
    input_size = 1
    output_size = 1
    print(f"input size = {input_size}")

    hidden_sizes = [5, 2]  # Количество нейронов в скрытых слоях

    activation_functions = [nn.Sigmoid, nn.Tanh, nn.ReLU]  # Список функций активации

    for activation in activation_functions:
        print(f"Training with activation function: {activation.__name__}")
        best_train, best_predictions = cross_validation(input_size, hidden_sizes, output_size,
                                                        activation, x_train, y_train,
                                                        cross_validation_count, epochs, learning_rate)
        # plot_classification_result(best_train, best_predictions, activation)


main()
