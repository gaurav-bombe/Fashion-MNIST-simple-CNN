import torch
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict

import helper

#  Define your network architecture here
input_size = 784
hidden_layer = [400, 250, 100]
output_layer = 10

model = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, hidden_layer[0])),
                        ('relu1', nn.ReLU()),
                        ('fc2', nn.Linear(hidden_layer[0], hidden_layer[1])),
                        ('relu2', nn.ReLU()),
                        ('fc3', nn.Linear(hidden_layer[1], hidden_layer[2])),
                        ('relu3', nn.ReLU()),
                        ('logits', nn.Linear(hidden_layer[2], output_layer))]))

print(model)

#  Create the network, define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

#  Train the network here
epochs = 3
print_no = 40
steps = 0
for i in range(epochs):
    running_loss = 0
    for images, labels in iter(trainloader):
        steps += 1

        images.resize_(images.size()[0], 784)
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_no == 0:
            print('Epoch {}/{}'.format(i + 1, epochs),
                  'Loss {:,.4f}'.format(running_loss / print_no))

            running_loss = 0

# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[3]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)
with torch.no_grad():
    logits = model.forward(img)
# Calculate the class probabilities (softmax) for img
print(logits.shape)
ps = F.softmax(logits, dim = 1)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')