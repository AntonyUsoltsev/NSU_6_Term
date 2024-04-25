import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
from Perceptron import Perceptron
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


def dataset_load():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    filtered_train_indices = [idx for idx, label in enumerate(train_dataset.targets) if
                              label.item() in classification_class]
    filtered_test_indices = [idx for idx, label in enumerate(test_dataset.targets) if
                             label.item() in classification_class]
    filtered_train_indices = filtered_train_indices[:500]

    train_dataset = Subset(train_dataset, filtered_train_indices)
    test_dataset = Subset(test_dataset, filtered_test_indices)

    train_loader = DataLoader(train_dataset, batch_size=250, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, test_loader


def draw_picture(train_loader):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(images.shape, labels.shape)
    f, axes = plt.subplots(1, 10, figsize=(30, 5))
    for i, axis in enumerate(axes):
        axes[i].imshow(np.squeeze(np.transpose(images[i].numpy(), (1, 2, 0))), cmap="gray")
        axes[i].set_title(labels[i].numpy())
    plt.show()


def main():
    epochs = 300
    learning_rate = 0.01

    K = 60  # K шагов градиентного спуска
    delta = 0.03  # Порог точности локального минимума

    train_loader, test_loader = dataset_load()
    draw_picture(train_loader)

    # Получение размера входных данных (количество пикселей в изображении)
    input_size = train_loader.dataset[0][0].shape[1] * train_loader.dataset[0][0].shape[2]

    train_data = []
    for images, labels in train_loader:
        # Преобразование данных в numpy массивы
        images = images.view(images.shape[0], -1).numpy()
        labels = labels.numpy().reshape(-1, 1)

        # Объединение изображений и меток
        train_data_batch = np.hstack((images, labels))
        train_data.append(train_data_batch)

    train_data = np.vstack(train_data)
    # train_data[:, -1] -= 1
    num_classes = len(classification_class)
    perceptron = Perceptron(feature_size=input_size + 1, num_classes=num_classes)  # +1 для учета смещения
    print(f"start train, train data: {train_data.shape}, num_classes: {num_classes}")
    # Обучаем перцептрон
    perceptron.sigm_classification(train_data, epochs, learning_rate, K, delta)
    print("end train")

    # Предсказываем метки для тестового набора
    predictions = []
    true_labels = []
    for images, labels in test_loader:
        # Преобразуем данные в numpy массивы
        images = images.view(images.shape[0], -1).numpy()
        labels = labels.numpy()

        # Предсказываем метки
        pred = perceptron.predict(images)
        predictions.extend(pred)
        true_labels.extend(labels)

    # Преобразуем списки в numpy массивы
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Оцениваем точность модели
    predicted_labels = [classification_class[i] for i in np.argmax(predictions, axis=1)]
    accuracy = np.mean(predicted_labels == true_labels)
    # print(predicted_labels)
    # print(true_labels)
    print(f'Accuracy: {accuracy}')


classification_class = [3]
main()
