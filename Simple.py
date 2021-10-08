import torch
import torch.nn as nn
from torch.nn import parameter
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.fc1 = nn.Linear(4704, 2000)
        self.fc2 = nn.Linear(2000, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1) # Flattens the data into a long vector
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
        



num_epochs = 10

device = torch.device('cpu')

# Loads the train and test data into PyTorch tensors
training_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())
test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())

# Loads the data into batches 
train_dataloader = DataLoader(training_data, batch_size = 100, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 100, shuffle = True)

model = SimpleNet().to(device)

loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_dataloader):
        data = data.to(device = device)
        labels = labels.to(device = device)

        #data = data.reshape(data.shape[0], -1) # Flattens the data into a long vector

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