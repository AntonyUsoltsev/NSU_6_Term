import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from ResNet import ResNet


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
    return train_loader, test_loader


def fit(train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_time_start = time.perf_counter()
    for epoch in range(10):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    train_time_end = time.perf_counter()
    print(f'Finished Training. Time = {(train_time_end - train_time_start):0.6f} sec')

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


def main():
    train_loader, test_loader = load_datasets()
    print("Start")
    fit(train_loader, test_loader)


main()
