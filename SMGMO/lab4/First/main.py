import matplotlib.pyplot as plt
import numpy as np
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


def ensemble_perceptrons(num_perceptrons, train_data, input_size, num_classes, epochs, learning_rate, K, delta):
    perceptrons = []
    for i in range(num_perceptrons):
        learning_rate -= 0.01
        perceptron = Perceptron(feature_size=input_size + 1, num_classes=num_classes)
        perceptron.sigm_classification(train_data, epochs, learning_rate, K, delta)
        perceptrons.append(perceptron)
        print(f"done for {i} perceptron")
    return perceptrons


def predict_ensemble(images, perceptrons):
    predictions = []
    for perceptron in perceptrons:
        pred = perceptron.predict(images)
        predictions.append(pred)
    ensemble_predictions = np.array(predictions).T
    final_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=ensemble_predictions)
    return final_predictions


def main():
    epochs = 100
    learning_rate = 0.10
    num_perceptrons = 3
    K = 40  # K шагов градиентного спуска
    delta = 0.02  # Порог точности локального минимума

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
    num_classes = len(classification_class)

    print(f"start train, train data: {train_data.shape}, num_classes: {num_classes}")
    perceptrons = ensemble_perceptrons(num_perceptrons, train_data, input_size, num_classes, epochs, learning_rate, K,
                                       delta)
    print("end train")

    predictions = []
    true_labels = []
    for images, labels in test_loader:
        images = images.view(images.shape[0], -1).numpy()
        labels = labels.numpy()
        pred = predict_ensemble(images, perceptrons)
        predictions.extend(pred)
        true_labels.extend(labels)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Оцениваем точность модели
    predicted_labels = [classification_class[i] for i in predictions]
    accuracy = np.mean(predicted_labels == true_labels)
    print(f'Accuracy: {accuracy}')


classification_class = [1, 8, 4]
main()
