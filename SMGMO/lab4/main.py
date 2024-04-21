import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


def dataset_load():
    # Определение преобразований данных
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразование в формат тензора
        transforms.Normalize((0.5,), (0.5,))  # Нормализация данных
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_dataset = Subset(train_dataset, range(1000))
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
    epochs = 100
    learning_rate = 0.03

    K = 50  # K шагов градиентного спуска
    delta = 0.01  # Порог точности локального минимума

    train_loader, test_loader = dataset_load()
    draw_picture(train_loader)

    # Получение размера входных данных (количество пикселей в изображении)
    input_size = train_loader.dataset[0][0].shape[1] * train_loader.dataset[0][0].shape[2]

    perceptron = Perceptron(feature_size=input_size + 1, num_classes=10)  # +1 для учета смещения
    i = 0
    train_data = []
    for images, labels in train_loader:
        # Преобразуем данные в numpy массивы
        images = images.view(images.shape[0], -1).numpy()
        labels = labels.numpy().reshape(-1, 1)

        # Объединяем изображения и метки
        train_data_batch = np.hstack((images, labels))
        train_data.append(train_data_batch)

    train_data = np.vstack(train_data)

    print(f"start train, train data: {train_data.shape}")
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
    # Convert probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f'Accuracy: {accuracy}')


main()
