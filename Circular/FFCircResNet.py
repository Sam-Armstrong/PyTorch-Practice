"""
Uses the pre-training circular model to train a single-layer perceptron for classifying MNIST digits.
"""

import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import time
import torch.optim.lr_scheduler as lr_s
import torchvision.transforms as transforms
from Block import Block
from CircResNet import CircResNet

class FFCircResNet(nn.Module):
    def __init__(self):
        super(FFCircResNet, self).__init__()

        self.fc = nn.Linear(4096, 10, bias = False)

        self.softmax = nn.Softmax(dim = 1)
        self.gelu = nn.GELU()
        self.relu = nn.GELU() #nn.ReLU()

        model = CircResNet()
        model.load_state_dict(torch.load('circ-resnet.pickle'))
        model.eval()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.blockset1 = model.blockset1
        self.blockset2 = model.blockset2
        self.blockset3 = model.blockset3
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.blockset1(x)
        x = self.blockset2(x)
        x = self.blockset3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def run_model():
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

        return (num_correct * 100 / num_samples).item()


    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    start_time = time.time()

    num_epochs = 20
    device = torch.device('cuda')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = transforms_train)
    test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = transforms_test)

    training_data, validation_set = random_split(training_data, [45000, 5000])

    # Loads the data into batches
    train_dataloader = DataLoader(training_data, batch_size = 500, shuffle = True)
    valid_dataloader = DataLoader(validation_set, batch_size = 500, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 500, shuffle = True)


    model = FFCircResNet().to(device)

    trainable_shapes = [torch.Size([10, 4096]), torch.Size([10])]

    # Prevents additional training or pre-trained layers
    for param in model.parameters():
        if param.shape in trainable_shapes:
            pass
        else:
            param.requires_grad = False

    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 1e-8)
    #scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 3)
    scheduler = lr_s.ExponentialLR(optimizer, gamma = 0.9)

    old_loss = 10000
    times_worse = 0

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            #print(batch_idx)
            #print(model.fc4.weight)
            #print(model.fc.weight[0])

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
            loss = loss_f(target, labels)
            valid_loss = loss.item() * data.size(0)

        scheduler.step()
        #print('Validation Loss: ', valid_loss)
        valid_accuracy = round(check_accuracy(valid_dataloader, model), 2)
        print(valid_accuracy, '% Validation Accuracy')

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

    print('\n')
    print(round(check_accuracy(test_dataloader, model), 2), '% Test Accuracy')

    print('Finished in %s seconds' % round(time.time() - start_time, 1))


if __name__ == '__main__':
    run_model()