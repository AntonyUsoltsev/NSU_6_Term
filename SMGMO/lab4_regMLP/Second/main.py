import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import MLP
from Generator import DataGenerator
from sklearn.model_selection import KFold


def draw_regression_result(x_t, y_t, x_p, y_p):
    x_f, y_f = generator.generate_function()
    plt.plot(x_f, y_f, label="Target Function")
    plt.plot(x_p, y_p, label="Predicted", color="red")
    plt.scatter(x_t, y_t, label="Dataset")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def train_model(model, x_train, y_train, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    error_func = nn.MSELoss()
    outputs = []
    for epoch in range(epochs):
        model.train()  # set model to training mode
        optimizer.zero_grad()  # zero the gradients
        outputs = model(x_train)  # forward pass
        loss = error_func(outputs.squeeze(), y_train)  # calculate the loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        model.eval()
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
    return outputs


def cross_validation(input_size, hidden_sizes, output_size, activation, x_data, y_data, k=5, epochs=1000, lr=0.3):
    kf = KFold(n_splits=k)
    losses = []
    min_loss = np.inf
    best_train = None
    best_predictions = None

    for train_index, test_index in kf.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        model = MLP(input_size, hidden_sizes, output_size, activation)
        train_model(model, x_train_tensor, y_train_tensor, epochs, lr)

        with torch.no_grad():
            outputs = model(x_test_tensor)
            loss = nn.MSELoss()(outputs.squeeze(), y_test_tensor)
            losses.append(loss)
            if loss < min_loss:
                min_loss = loss
                # best_train = np.concatenate((x_train, x_test))
                # best_predictions = np.concatenate((model_predictions.numpy(), outputs.numpy()))
                best_train = np.arange(-1, 1, 0.01)
                contin_outputs = model(torch.tensor(np.arange(-1, 1, 0.01), dtype=torch.float32).unsqueeze(1))
                best_predictions = contin_outputs.numpy()
            # draw_regression_result(x_train,
            #                        y_train,
            #                        np.arange(-1, 1, 0.01),
            #                        model(
            #                            torch.tensor(np.arange(-1, 1, 0.01), dtype=torch.float32).unsqueeze(1)).numpy())

        print("Cross validation iteration end\n")

    avg_loss = np.mean(losses)
    print(f'Average loss across {k}-fold cross-validation: {avg_loss}')
    print(f'Min loss across {k}-fold cross-validation: {min_loss}\n')

    return best_train, best_predictions


def main():
    num_points = 100
    noise = 0.0
    epochs = 2000
    learning_rate = 0.003
    cross_validation_count = 5
    x_train, y_train = generator.generate(num_points, noise)

    # Задание параметров модели и обучение
    input_size = 1
    output_size = 1
    hidden_sizes = [7, 8, 5]  # Количество нейронов в скрытых слоях
    # hidden_sizes = [10, 15, 5, 4]  # Количество нейронов в скрытых слоях
    activation_functions = [nn.Sigmoid, nn.Tanh, nn.ReLU]  # Список функций активации

    for activation in activation_functions:
        print(f"Training with activation function: {activation.__name__}")
        best_train, best_predictions = cross_validation(input_size, hidden_sizes, output_size,
                                                        activation, x_train, y_train,
                                                        cross_validation_count, epochs, learning_rate)
        draw_regression_result(x_train, y_train, best_train, best_predictions)


function_type = "sin"
poly_coeffs = [-4, 2, 7, -2]
generator = DataGenerator(function_type, poly_coeffs)

main()
