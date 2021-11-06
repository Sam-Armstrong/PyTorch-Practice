"""
Author: Sam Armstrong
Date: November 2021

Description: Implementation of the Circular neural network architecture that uses dense rather than sparse layers,
and uses separate network layers for the full pass, rather than reversing the layers.
"""

import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import time
import torch.optim.lr_scheduler as lr_s
from Sparse import Sparse

class NewCircular(nn.Module):
    def __init__(self):
        super(NewCircular, self).__init__()
        self.sp = Sparse(784, 784, connections_per_neuron = 2)
        self.fc = nn.Linear(784, 10)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
        self.bn1 = nn.BatchNorm1d(1500)
        self.bn2 = nn.BatchNorm1d(2500)
        self.bn3 = nn.BatchNorm1d(2500)
        self.bn4 = nn.BatchNorm1d(1500)

        self.dropout = nn.Dropout(p = 0.5)

        self.fc1 = nn.Linear(784, 1500)
        self.fc2 = nn.Linear(1500, 2500)
        self.fc3 = nn.Linear(2500, 2500)
        self.fc4 = nn.Linear(2500, 1500)
        self.fc5 = nn.Linear(1500, 784)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1) # Flattens data

        # Drops out an amount of the input
        x = self.dropout(x)

        # Propagating forwards into the network
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.bn2(x)

        # Self-comparison layer (central layer training with relation to itself)
        x = self.fc3(x)
        x = self.gelu(x)
        x = self.bn3(x)
        
        # Neuron gradients then propagate back through the network to the input layer
        x = self.fc4(x)
        x = self.gelu(x)
        x = self.bn4(x)
        x = self.fc5(x)
        x = self.relu(x) # Last activation is relu to better approximate the input values

        return x 


def train_model():
    start_time = time.time()

    num_epochs = 10
    device = torch.device('cpu')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.MNIST(root = "data", train = True, download = True, transform = ToTensor())
    test_data = datasets.MNIST(root = "data", train = False, download = True, transform = ToTensor())
    training_data, validation_set = random_split(training_data,[50000,10000])

    # Loads the data into batches
    train_dataloader = DataLoader(training_data, batch_size = 200, shuffle = True)
    valid_dataloader = DataLoader(validation_set, batch_size = 200, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 200, shuffle = True)

    model = NewCircular().to(device)

    params = []
    params += model.parameters()

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(params, lr = 0.00001, weight_decay = 0)
    scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 2)

    old_loss = 10000
    times_worse = 0

    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            #print(batch_idx)

            batch_size = data.shape[0]
            data = data.to(device = device)
            flattened_data = data.reshape(batch_size, 784).clone().detach()

            scores = model(data) # Runs a forward pass of the model for all the data
            loss = loss_f(scores, flattened_data) # Calculates the loss of the forward pass using the loss function
            #print('Training Loss: ', loss.item())
            train_loss += loss.item()

            optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
            loss.backward() # Backpropagates the network using the loss to calculate the local gradients

            optimizer.step() # Updates the network weights and biases

        print('Training Loss: ', train_loss)


    torch.save(model.state_dict(), 'new-circular.pickle')
    print('Circular saved to .pickle file')
    print('Finished in %s seconds' % round(time.time() - start_time, 1))


if __name__ == '__main__':
    train_model()
