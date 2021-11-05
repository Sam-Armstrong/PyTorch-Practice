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

class CircularNN(nn.Module):
    def __init__(self):
        super(CircularNN, self).__init__()
        self.sp = Sparse(784, 784, connections_per_neuron = 2)
        self.fc = nn.Linear(784, 10)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
        self.bn1 = nn.BatchNorm1d(2500)
        self.bn2 = nn.BatchNorm1d(2500)
        self.bn3 = nn.BatchNorm1d(1500)

        self.fc1 = nn.Linear(784, 1500)
        self.fc2 = nn.Linear(1500, 2500)
        self.fc3 = nn.Linear(2500, 2500)
        self.fc4 = nn.Linear(2500, 1500)
        self.fc5 = nn.Linear(1500, 784)

    def forward(self, x):
        ## Remember to think about the input order (especially when the input is an image)

        # Propagating forwards into the network
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.bn1(x)

        # Self-comparison layer (central layer training with relation to itself)
        x = self.fc3(x)
        x = self.gelu(x)
        x = self.bn2(x)
        
        # Neuron gradients then propagate back through the network to the input layer
        x = self.fc4(x)
        x = self.gelu(x)
        x = self.bn3(x)
        x = self.fc5(x)
        x = self.relu(x) # Last activation is relu to better approximate the input values

        return x 


def train_model():
    start_time = time.time()

    num_epochs = 1
    device = torch.device('cpu')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())
    test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())
    training_data, validation_set = random_split(training_data,[50000,10000])

    # Loads the data into batches
    train_dataloader = DataLoader(training_data, batch_size = 200, shuffle = True)
    valid_dataloader = DataLoader(validation_set, batch_size = 200, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 200, shuffle = True)

    model = CircularNN().to(device)

    params = []
    params += model.parameters()

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(params, lr = 0.001, weight_decay = 0)
    scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 2)

    old_loss = 10000
    times_worse = 0

    for epoch in range(num_epochs):
        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            print(batch_idx)

            data = data.to(device = device)
            #labels = labels.to(device = device)
            
            batch_size = data.shape[0]
            new_data = data.reshape(batch_size, 784)
            flattened_data = new_data.clone().detach()

            # For k missing values
            k = 150
            random_tensor = torch.rand((batch_size, 784))
            _, left_out_indices = random_tensor.topk(k)
            #new_labels = torch.empty((batch_size, k)) # Currently not using the labels tensor (not required for training)

            #new_data[left_out_indices] = 0 # Removes the missing inputs from the data
            for i, x in enumerate(left_out_indices):
                for j, y in enumerate(x):
                    y_item = y.item()
                    new_data[i, y_item] = 0


            ## new_labels is an array of the single missing input for each sample in the batch
            ## new_data is the same input data (flattened), but missing a single value, which is now 0

            ### Currently the model is trained with the labels being the entire input sample; Change this to just the missing value??

            scores = model(new_data) # Runs a forward pass of the model for all the data
            loss = loss_f(scores, flattened_data) # Calculates the loss of the forward pass using the loss function
            print('Loss: ', loss.item())
            train_loss += loss

            optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
            loss.backward() # Backpropagates the network using the loss to calculate the local gradients

            optimizer.step() # Updates the network weights and biases


    # Checks the performance of the model on the test set
    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_dataloader):
                print(batch_idx)

                data = data.to(device = device)
                batch_size = data.shape[0]
                new_data = data.reshape(batch_size, 784)

                # For k missing values
                k = 150
                random_tensor = torch.rand((batch_size, 784))
                _, left_out_indices = random_tensor.topk(k)
                new_labels = torch.empty((batch_size, k))

                #new_data[left_out_indices] = 0 # Removes the missing inputs from the data
                for i, x in enumerate(left_out_indices):
                    for j, y in enumerate(x):
                        y_item = y.item()
                        new_labels[i, j] = new_data[i, y_item].item()
                        new_data[i, y_item] = 0

                scores = model(new_data)

                for i in range(batch_size):
                    for j in range(k):
                        if -0.005 < new_labels[i][j] - scores[i][left_out_indices[i, j]] < 0.005:
                            num_correct += 1

                num_samples += batch_size * k

        print((num_correct * 100 / num_samples), '%  Correct')

    print('\n')
    print('Training finished in %s seconds' % round(time.time() - start_time, 1))

    torch.save(model.state_dict(), 'circular-model.pickle')
    print('Circular saved to .pickle file')

    print('Checking accuracy...\n')
    check_accuracy(test_dataloader, model)

    print('Finished in %s seconds' % round(time.time() - start_time, 1))


if __name__ == '__main__':
    train_model()
