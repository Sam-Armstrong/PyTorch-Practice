"""
Author: Sam Armstrong (based upon the research of (currently unknown author))
Date: 13/10/2021
Decription: PyTorch re-implementation of the 'ConvMixer' deep learning architecture that was described in the recent 
paper 'Patches are all you need?' (2021)
"""

import torch
import torch.nn as nn
from Residual import Residual
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size = 7, patch_size = 4, n_classes = 10):
        super(ConvMixer, self).__init__()
        self.depth = depth

        self.conv1 = nn.Conv2d(1, dim, kernel_size = patch_size, stride = patch_size)
        self.gelu = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(dim)
        self.residuals = []
        self.conv2s = []
        for i in range(depth):
            self.residuals.append(Residual(dim, kernel_size))
            self.conv2s.append(nn.Conv2d(dim, dim, kernel_size = 1))

        self.conv2 = nn.Conv2d(dim, dim, kernel_size = 1)
        self.residual = Residual(dim, kernel_size)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.batch_norm(x)

        x = self.residual(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.batch_norm(x)

        """x_matrix = torch.zeros((x.shape))

        for i in range(self.depth):
            x = self.residuals[i](x)
            x = self.conv2s[i](x)
            x = self.gelu(x)
            x = self.batch_norm(x)
            x_matrix += x"""
        
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x




num_epochs = 10

device = torch.device('cpu')

# Loads the train and test data into PyTorch tensors
training_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())
test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())

# Loads the data into batches
train_dataloader = DataLoader(training_data, batch_size = 100, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 100, shuffle = True)

model = ConvMixer(1, 1).to(device)

loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_dataloader):
        data = data.to(device = device)
        labels = labels.to(device = device)

        #data = data.reshape(data.shape[0], -1) # Flattens the data into a vector

        scores = model(data) # Runs a forward pass of the model for all the data
        loss = loss_f(scores, labels) # Calculates the loss of the forward pass using the loss function

        optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
        loss.backward() # Backpropagates the network using the loss to calculate the local gradients

        optimizer.step() # Updates the network weights and biases


# Checks the performance of the model on the test set
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print((num_correct * 100 / num_samples).item(), '%  Correct')


check_accuracy(test_dataloader, model)