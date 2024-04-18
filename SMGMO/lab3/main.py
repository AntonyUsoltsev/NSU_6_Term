import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import Generator as generator

from Perceptron import MLP


def plot_classification_result(model, x_data, y_data):
    # Преобразование тензоров в numpy массивы
    # x_data = x_data.cpu().detach().numpy()
    # y_data = y_data.cpu().detach().numpy()

    # Вычисление предсказаний модели
    with torch.no_grad():
        outputs = model(x_data)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()

    # Разделение данных по классам
    class_0_x = x_data[predictions == 0]
    class_0_y = y_data[predictions == 0]
    class_1_x = x_data[predictions == 1]
    class_1_y = y_data[predictions == 1]

    # Построение графика
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
def train_model(model, x_train, y_train, epochs, lr=0.3):
    error_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        outputs = model(x_train).squeeze()

        loss = error_func(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            XXT = x_train.clone().detach()
            test_pred = model(XXT).squeeze()
            YYT = y_train.clone().detach()
            test_error = error_func(test_pred, YYT)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss}, Test error {test_error}')


def main():
    num_points = 40
    noise = 0.0
    epochs = 100
    # Задание данных для обучения
    train_set = generator.generate(num_points, noise, "gauss")
    draw_set(train_set)
    # Преобразование данных в тензоры PyTorch
    x_train = train_set[:, :2]
    y_train = train_set[:, 2]
    # print(x_train)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Задание параметров модели и обучение
    input_size = x_train.shape[1]
    output_size = len(np.unique(y_train))  # Количество классов
    print(f"input size = {input_size}, output size = {output_size}")
    hidden_sizes = [5, 3, 2]  # Пример количества нейронов в скрытых слоях
    activation_functions = [nn.Sigmoid, nn.Tanh, nn.ReLU]  # Список функций активации

    for activation in activation_functions:
        print(f"Training with activation function: {activation.__name__}")
        model = MLP(input_size, hidden_sizes, output_size, activation)
        train_model(model, x_train_tensor, y_train_tensor, epochs)
        plot_classification_result(model, x_train_tensor, y_train_tensor)


main()
