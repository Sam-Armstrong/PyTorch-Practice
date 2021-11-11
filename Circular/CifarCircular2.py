"""
Author: Sam Armstrong
Date: November 2021

Description:
"""

import torch.nn as nn
import torch
from torch.nn.modules import flatten
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import time
import torch.optim.lr_scheduler as lr_s

class CifarCircular2(nn.Module):
    def __init__(self):
        super(CifarCircular2, self).__init__()
        #self.fc = nn.Linear(784, 10)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

        self.bn1 = nn.BatchNorm1d(5000)
        self.bn2 = nn.BatchNorm1d(10000)
        self.bn3 = nn.BatchNorm1d(10000)
        self.bn4 = nn.BatchNorm1d(5000)

        self.convbn1 = nn.BatchNorm2d(9)
        self.convbn2 = nn.BatchNorm2d(18)
        self.convbn3 = nn.BatchNorm2d(18)
        self.convbn4 = nn.BatchNorm2d(18)

        self.dropout = nn.Dropout(p = 0.5)

        self.conv1 = nn.Conv2d(3, 9, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(9, 18, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(18, 18, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(18, 18, kernel_size = 3, padding = 1)

        self.pooling = nn.MaxPool2d(kernel_size = 2)

        self.conv10 = nn.Conv2d(3, 6, kernel_size = 3, padding = 1)
        self.conv11 = nn.Conv2d(6, 3, kernel_size = 1, padding = 0)

        self.fc1 = nn.Linear(18432, 5000) # 18432 # 4608
        self.fc2 = nn.Linear(5000, 10000)
        self.fc3 = nn.Linear(10000, 10000)
        self.fc4 = nn.Linear(10000, 5000)
        self.fc5 = nn.Linear(5000, 3072)

    def forward(self, x):
        #x = x.reshape(x.shape[0], -1) # Flattens data

        # Drops out an amount of the input
        x = self.dropout(x)

        # Propagating forwards into the network
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.convbn1(x)

        x = self.conv2(x)
        x = self.gelu(x)
        x = self.convbn2(x)

        #x = self.pooling(x)

        x = self.conv3(x)
        x = self.gelu(x)
        x = self.convbn3(x)

        x = self.conv4(x)
        x = self.gelu(x)
        x = self.convbn4(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.gelu(x)
        #x = self.bn1(x)

        x = self.fc2(x)
        x = self.gelu(x)
        #x = self.bn2(x)

        # Self-comparison layer (central layer training with relation to itself)
        x = self.fc3(x)
        x = self.gelu(x)
        #x = self.bn3(x)
        
        # Neuron gradients then propagate back through the network to the input layer
        x = self.fc4(x)
        x = self.gelu(x)
        #x = self.bn4(x)

        x = self.fc5(x)
        x = self.gelu(x)

        x = x.reshape(x.shape[0], 3, 32, 32)

        x = self.conv10(x)
        x = self.gelu(x)

        x = self.conv11(x)
        x = self.relu(x) # Last activation is relu to better approximate the input values

        x = x.reshape(x.shape[0], -1)

        return x


def train_model():
    start_time = time.time()

    num_epochs = 20
    device = torch.device('cuda')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = ToTensor())
    test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = ToTensor())
    training_data, validation_set = random_split(training_data, [45000, 5000])

    # Loads the data into batches
    train_dataloader = DataLoader(training_data, batch_size = 500, shuffle = True)
    valid_dataloader = DataLoader(validation_set, batch_size = 500, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True)

    model = CifarCircular2().to(device)

    params = []
    params += model.parameters()

    loss_f = nn.MSELoss()
    optimizer = optim.AdamW(params, lr = 0.0001, weight_decay = 0)
    scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 3)

    old_loss = 10000
    times_worse = 0

    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            #print(batch_idx)

            batch_size = data.shape[0]
            data = data.to(device = device)
            flattened_data = data.reshape(batch_size, 3072).clone().detach()

            scores = model(data) # Runs a forward pass of the model for all the data
            loss = loss_f(scores, flattened_data) # Calculates the loss of the forward pass using the loss function
            #print('Training Loss: ', loss.item())
            train_loss += loss.item()

            optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
            loss.backward() # Backpropagates the network using the loss to calculate the local gradients

            optimizer.step() # Updates the network weights and biases

        print('Training Loss: ', train_loss)

        valid_loss = 0.0
        model.eval()
        for data, labels in valid_dataloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            target = model(model.dropout(data))

            flattened = data.clone().detach().reshape(data.shape[0], -1)
            loss = loss_f(target, flattened)
            valid_loss = loss.item() * data.size(0)

        scheduler.step(valid_loss)
        print('Validation Loss: ', valid_loss)

        if valid_loss >= old_loss:
            times_worse += 1

        else:
            times_worse = 0


        if times_worse >= 3:
            print('Reducing learning rate.')

        if times_worse >= 6:
            print('Stopping early.')
            start_time = time.time()
            break

        old_loss = valid_loss


    torch.save(model.state_dict(), 'cifar-circular.pickle')
    print('Circular saved to .pickle file')
    print('Finished in %s seconds' % round(time.time() - start_time, 1))


if __name__ == '__main__':
    train_model()
