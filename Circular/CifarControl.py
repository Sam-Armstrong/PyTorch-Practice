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

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def run_model():
    start_time = time.time()

    num_epochs = 10
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
    optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay = 0)
    scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 2)

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


        if times_worse >= 2:
            print('Reducing learning rate.')

        if times_worse >= 3:
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