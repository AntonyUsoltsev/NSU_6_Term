import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import Generator as generator
from Perceptron import MLP
from sklearn.model_selection import KFold


def plot_classification_result(x_data, predictions):
    class_0_x = x_data[predictions == 0]
    class_1_x = x_data[predictions == 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(class_0_x[:, 0], class_0_x[:, 1], color='red', label='Class 0')
    plt.scatter(class_1_x[:, 0], class_1_x[:, 1], color='blue', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Classification Result')
    plt.legend()
    plt.show()


def draw_set(set_to_draw):
    class_0 = set_to_draw[set_to_draw[:, 2] == 0]
    class_1 = set_to_draw[set_to_draw[:, 2] == 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', alpha=0.5)
    plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 1', alpha=0.5)
    plt.show()


# Функция для обучения модели
def train_model(model, x_train, y_train, epochs, lr):
    error_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    outputs = []
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()  # обнуление весов

        outputs = model(x_train)  # forward pass

        # print(f"train: {y_train.shape}, pred: {outputs.shape}")
        loss = error_func(outputs, y_train)  # вычисление ошибки

        loss.backward()  # back propagation - вычисление градиентов

        optimizer.step()  # обновление весов сети

        model.eval()
        with torch.inference_mode():
            XXT = x_train.clone().detach()
            test_pred = model(XXT).squeeze()
            YYT = y_train.clone().detach()
            test_error = error_func(test_pred, YYT)

        if (epoch + 1) % 250 == 0:
            accuracy = accuracy_score(y_train.numpy(), np.argmax(model.forward(x_train).detach().numpy(), axis=1))
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss}, accuracy {accuracy}, test error: {test_error}')
    return outputs

def cross_validation(model, x_data, y_data, k=5, epochs=1000, lr=0.3):
    kf = KFold(n_splits=k)
    accuracies = []
    max_accuracy = 0
    best_predictions = None

    for train_index, test_index in kf.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        model_predictions = train_model(model, x_train_tensor, y_train_tensor, epochs, lr)

        with torch.no_grad():
            outputs = model(x_test_tensor)
            _, predictions = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
            accuracies.append(accuracy)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_predictions = predictions.numpy()

    avg_accuracy = np.mean(accuracies)
    print(f'Average accuracy across {k}-fold cross-validation: {avg_accuracy}')
    print(f'Max accuracy across {k}-fold cross-validation: {max_accuracy}')

    return best_predictions


def main():
    num_points = 100
    noise = 0.0
    epochs = 1000
    learning_rate = 0.3
    # Задание данных для обучения
    train_set = generator.generate(num_points, noise, "circle")
    draw_set(train_set)
    # Преобразование данных в тензоры PyTorch
    x_train = train_set[:, :2]
    y_train = train_set[:, 2]
    # print(x_train)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Задание параметров модели и обучение
    input_size = x_train.shape[1]
    output_size = len(np.unique(y_train))  # Количество классов
    print(f"input size = {input_size}, output size = {output_size}")
    hidden_sizes = [5, 3, 2]  # Пример количества нейронов в скрытых слоях
    activation_functions = [nn.Sigmoid, nn.Tanh, nn.ReLU]  # Список функций активации

    for activation in activation_functions:
        print(f"Training with activation function: {activation.__name__}")
        model = MLP(input_size, hidden_sizes, output_size, activation)
        best_predictions = cross_validation(model, x_train, y_train, 5, epochs, learning_rate)
        plot_classification_result(x_train, best_predictions)


main()
