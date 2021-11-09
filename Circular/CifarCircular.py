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

class CifarCircular(nn.Module):
    def __init__(self):
        super(CifarCircular, self).__init__()
        #self.fc = nn.Linear(784, 10)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        #self.bn1 = nn.BatchNorm1d(1500)
        #self.bn2 = nn.BatchNorm1d(2000)
        #self.bn3 = nn.BatchNorm1d(2000)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(6)
        self.bn3 = nn.BatchNorm2d(6)
        self.bn4 = nn.BatchNorm2d(12)
        self.bn5 = nn.BatchNorm2d(12)
        self.bn6 = nn.BatchNorm2d(12)

        self.bn7 = nn.BatchNorm2d(12)
        self.bn8 = nn.BatchNorm2d(12)
        self.bn9 = nn.BatchNorm2d(6)
        self.bn10 = nn.BatchNorm2d(6)
        self.bn11 = nn.BatchNorm2d(6)

        self.dropout = nn.Dropout(p = 0.5)

        self.fc1 = nn.Linear(768, 3072)

        self.conv1 = nn.Conv2d(3, 6, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(6, 6, kernel_size = 2, padding = 0, stride = 2) # Patch encoding
        self.conv4 = nn.Conv2d(6, 12, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(12, 12, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(12, 12, kernel_size = 2, padding = 0, stride = 2) # Patch encoding

        self.deconv1 = nn.ConvTranspose2d(12, 12, kernel_size = 2, padding = 0, stride = 2)
        self.deconv2 = nn.ConvTranspose2d(12, 12, kernel_size = 3, padding = 1)
        self.deconv3 = nn.ConvTranspose2d(12, 6, kernel_size = 3, padding = 1)
        self.deconv4 = nn.ConvTranspose2d(6, 6, kernel_size = 2, padding = 0, stride = 2)
        self.deconv5 = nn.ConvTranspose2d(6, 6, kernel_size = 3, padding = 1)
        self.deconv6 = nn.ConvTranspose2d(6, 3, kernel_size = 3, padding = 1)

        #self.conv7 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        #self.conv8 = nn.Conv2d(3, 3, kernel_size = 1, padding = 0)

    def forward(self, x):
        #x = x.reshape(x.shape[0], -1) # Flattens data

        # Drops out an amount of the input
        x = self.dropout(x)

        # Propagating forwards into the network
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.gelu(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.gelu(x)
        x = self.bn4(x)
        
        x = self.conv5(x)
        x = self.gelu(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.gelu(x)
        x = self.bn6(x)

        x = self.deconv1(x)
        x = self.gelu(x)
        x = self.bn7(x)

        x = self.deconv2(x)
        x = self.gelu(x)
        x = self.bn8(x)

        x = self.deconv3(x)
        x = self.gelu(x)
        x = self.bn9(x)

        x = self.deconv4(x)
        x = self.gelu(x)
        x = self.bn10(x)
        
        x = self.deconv5(x)
        x = self.gelu(x)
        x = self.bn11(x)

        x = self.deconv6(x)

        #x = x.reshape(x.shape[0], -1)

        #x = self.fc1(x)
        #x = self.gelu(x)

        #x = x.reshape(x.shape[0], 3, 32, 32)
        

        """x = self.conv7(x)
        x = self.gelu(x)

        x = self.conv8(x)"""
        
        x = self.relu(x) # Last activation is relu to better approximate the input values

        x = x.reshape(x.shape[0], -1)

        return x


def train_model():
    start_time = time.time()

    num_epochs = 30
    device = torch.device('cpu')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = ToTensor())
    test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = ToTensor())
    training_data, validation_set = random_split(training_data, [45000, 5000])

    # Loads the data into batches
    train_dataloader = DataLoader(training_data, batch_size = 200, shuffle = True)
    valid_dataloader = DataLoader(validation_set, batch_size = 200, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 200, shuffle = True)

    model = CifarCircular().to(device)

    params = []
    params += model.parameters()

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(params, lr = 0.00003, weight_decay = 0)
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
            
            target = model(data)

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
