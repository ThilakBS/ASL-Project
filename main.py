import os
import copy
import time
import torch
import torchvision
import matplotlib
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from CNN_CLASS import Net
from torch.utils.data import random_split, DataLoader


PATH = './sign_lang.pth'


transform = transforms.Compose(
    [transforms.Resize((200, 200)), # Resize the images to 224x224 pixels
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5190, 0.4977, 0.5134), std=(0.2028, 0.2328, 0.2416))])

path = r"C:\Users\hbala\Downloads\FULLDATASET\asl_alphabet_train\asl_alphabet_train"
dataset = datasets.ImageFolder(root=path, transform=transform)

# Determine the lengths of split
train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len

# Random split
train_dataset, test_dataset = random_split(dataset, lengths=[train_len, test_len])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)




device = torch.device("cpu") # Use CPU, since you mentioned you don't have a GPU

# Initialize your custom model

print(time.time())
for epoch in range(2):
    net = Net()

    net = net.to(device)
    if os.path.isfile(PATH):
        net.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    st= time.time()


    print(epoch)

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(running_loss)
    et = time.time()
    print(st - et)

    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the validation images: {100 * correct // total} %')
    print('Finished Training')

    torch.save(net.state_dict(), PATH)

