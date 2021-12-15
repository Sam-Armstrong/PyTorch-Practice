from math import gamma
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

class CircResNet(nn.Module):
    def __init__(self):
        super(CircResNet, self).__init__()

        self.relu = nn.GELU() #nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(p = 0.5)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, stride = 1, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)

        self.blockset1 = nn.Sequential(
            Block(16, 16, stride = 1),
            Block(16, 16, stride = 1),
            Block(16, 16, stride = 1),
            Block(16, 16, stride = 1),
            Block(16, 16, stride = 1)
        )

        self.blockset2 = nn.Sequential(
            Block(16, 32, stride = 2),
            Block(32, 32, stride = 1),
            Block(32, 32, stride = 1),
            Block(32, 32, stride = 1),
            Block(32, 32, stride = 1)
        )

        self.blockset3 = nn.Sequential(
            Block(32, 64, stride = 2),
            Block(64, 64, stride = 1),
            Block(64, 64, stride = 1),
            Block(64, 64, stride = 1),
            Block(64, 64, stride = 1)
        )

        #self.pool = nn.AvgPool2d(kernel_size = 8, stride = 1)
        # self.conv2 = nn.Conv2d(64, 3, kernel_size = 1, stride = 1, padding = 0, bias = False)
        # self.bn2 = nn.BatchNorm2d(3)
        self.fc = nn.Linear(4096, 3072)

        for m in self.modules():
            #print(m)
            if isinstance(m, Block):
                nn.init.kaiming_normal_(m.conv1.weight, mode = 'fan_out', nonlinearity = 'relu')
                nn.init.kaiming_normal_(m.conv2.weight, mode = 'fan_out', nonlinearity = 'relu')
                nn.init.constant_(m.bn1.weight, 1)
                nn.init.constant_(m.bn1.bias, 0)
                nn.init.constant_(m.bn2.weight, 1)
                nn.init.constant_(m.bn2.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.blockset1(x)
        x = self.blockset2(x)
        x = self.blockset3(x)

        # Current Shape (500, 64, 8, 8) # 4096
        # Shape needed (500, 3, 32, 32) # 3072

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        #x = x.reshape(x.shape[0], 3, 32, 32)
        #x = self.conv2(x)
        x = self.relu(x)
        #x = x.reshape(x.shape[0], -1)
        return x



def run_model():

    # transforms_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # transforms_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])


    start_time = time.time()

    num_epochs = 10
    device = torch.device('cuda')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = ToTensor()) #transforms_train
    #test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = ToTensor()) # Not used # transforms_test

    training_data, validation_set = random_split(training_data, [45000, 5000])

    # Loads the data into batches
    train_dataloader = DataLoader(training_data, batch_size = 500, shuffle = True)
    valid_dataloader = DataLoader(validation_set, batch_size = 500, shuffle = True)

    model = CircResNet().to(device)

    params = []
    params += model.parameters()

    loss_f = nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr = 1e-2, weight_decay = 1e-7) # Best lr = 0.001 # weight_decay = 1e-6

    #scheduler = lr_s.MultiStepLR(optimizer, milestones = [20, 80], gamma = 0.1)
    scheduler = lr_s.ExponentialLR(optimizer, gamma = 0.9)

    old_loss = 10000
    times_worse = 0

    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        train_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            #print(batch_idx)

            data = data.to(device = device)
            labels = labels.to(device = device)
            flattened = data.clone().detach().reshape(data.shape[0], -1)

            scores = model(data) # Runs a forward pass of the model for all the data
            loss = loss_f(scores, flattened) # Calculates the loss of the forward pass using the loss function
            train_loss += loss

            optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
            loss.backward() # Backpropagates the network using the loss to calculate the local gradients

            optimizer.step() # Updates the network weights and biases


        valid_loss = 0.0
        model.eval()
        for data, labels in valid_dataloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            target = model(model.dropout(data))
            loss = loss_f(target, flattened)
            valid_loss = loss.item() * data.size(0)

        scheduler.step()

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


    print('\n')

    torch.save(model.state_dict(), 'circ-resnet.pickle')
    print('Saved to .pickle file')

    print('Finished in %s seconds' % round(time.time() - start_time, 1))

if __name__ == '__main__':
    run_model()