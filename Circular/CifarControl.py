"""
A control model to test whether the models I have developed create any improvement in accuracy
"""

import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import time
import torch.optim.lr_scheduler as lr_s

class ControlModel(nn.Module):
    def __init__(self):
        super(ControlModel, self).__init__()
        self.fc = nn.Linear(3072, 10)
        self.softmax = nn.Softmax(dim = 1)

        self.conv1 = nn.Conv2d(3, 6, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(6, 6, kernel_size = 2, padding = 0, stride = 2) # Patch encoding
        self.conv4 = nn.Conv2d(6, 12, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(12, 12, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(12, 12, kernel_size = 2, padding = 0, stride = 2) # Patch encoding
        
        self.conv7 = nn.Conv2d(12, 12, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(12, 12, kernel_size = 3, padding = 1)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(6)
        self.bn3 = nn.BatchNorm2d(6)
        self.bn4 = nn.BatchNorm2d(12)
        self.bn5 = nn.BatchNorm2d(12)
        self.bn6 = nn.BatchNorm2d(12)
        self.bn7 = nn.BatchNorm2d(12)
        self.bn8 = nn.BatchNorm2d(12)
        self.bn9 = nn.BatchNorm1d(400)
        self.bn10 = nn.BatchNorm1d(100)

        self.pooling1 = nn.MaxPool2d(kernel_size = 2)
        self.pooling2 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.gelu(x)
        x = self.bn2(x)

        # x = self.conv3(x)
        # x = self.gelu(x)
        # x = self.bn3(x)

        x = self.pooling1(x)

        x = self.conv4(x)
        x = self.gelu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.gelu(x)
        x = self.bn5(x)

        # x = self.conv6(x)
        # x = self.gelu(x)
        # x = self.bn6(x)

        x = self.pooling2(x)
        
        x = self.conv7(x)
        x = self.gelu(x)
        x = self.bn7(x)

        x = self.conv8(x)
        x = self.gelu(x)
        x = self.bn8(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.bn9(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.bn10(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x


def run_model():
    start_time = time.time()

    num_epochs = 10
    
    if torch.cuda.is_available():
        print('Cuda is available!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = ToTensor())
    test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = ToTensor())

    training_data, validation_set = random_split(training_data, [45000, 5000])

    # Loads the data into batches 
    train_dataloader = DataLoader(training_data, batch_size = 200, shuffle = True)
    valid_dataloader = DataLoader(validation_set, batch_size = 200, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 200, shuffle = True)


    model = ControlModel().to(device)

    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.00003, weight_decay = 0)
    scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 3)

    old_loss = 10000
    times_worse = 0

    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            #print(batch_idx)
            data = data.to(device = device)
            labels = labels.to(device = device)

            scores = model(data) # Runs a forward pass of the model for all the data
            loss = loss_f(scores, labels) # Calculates the loss of the forward pass using the loss function
            train_loss += loss

            optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
            loss.backward() # Backpropagates the network using the loss to calculate the local gradients

            optimizer.step() # Updates the network weights and biases


        valid_loss = 0.0
        model.eval()
        for data, labels in valid_dataloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            target = model(data)
            loss = loss_f(target,labels)
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

    print('\n')
    check_accuracy(test_dataloader, model)

    print('Finished in %s seconds' % round(time.time() - start_time, 1))


if __name__ == '__main__':
    run_model()