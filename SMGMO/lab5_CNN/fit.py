import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from CNN import CNN
from ConvBlocks import ConvBlockA, ConvBlockB


def load_datasets():
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(),
                                       target_transform=None)
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(),
                                      target_transform=None)

    train_data_crop = Subset(train_data, range(3000))
    test_data_crop = Subset(test_data, range(500))

    class_names = train_data.classes
    class_to_idx = train_data.class_to_idx
    print(class_names, class_to_idx)
    image, label = train_data[0]
    img_size = image.squeeze().size()[0]
    print(image.squeeze().size())

    torch.manual_seed(20)
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        img, label = train_data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False)
    plt.show()

    train_loader = DataLoader(dataset=train_data_crop, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data_crop, batch_size=64, shuffle=False)
    return train_loader, test_loader, img_size


def fit(train_loader, test_loader, img_size, conv_block):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(conv_block, num_classes=10, img_size=img_size)
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Оценка модели на тестовом датасете
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    precision = precision_score(true_labels, predicted_labels, average='macro')
    print(f'Precision of the network on the test images: {precision:.2f}')

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')


def main():
    train_loader, test_loader, img_size = load_datasets()

    # print("Start with ConvBlockA")
    # fit(train_loader, test_loader, img_size,ConvBlockA)
    print("Start with ConvBlockB")
    fit(train_loader, test_loader, img_size, ConvBlockB)


main()
