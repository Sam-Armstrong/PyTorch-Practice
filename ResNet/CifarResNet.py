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

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

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

        self.pool = nn.AvgPool2d(kernel_size = 8, stride = 1)
        self.fc = nn.Linear(64, 10)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.blockset1(x)
        x = self.blockset2(x)
        x = self.blockset3(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x) # Works best with no softmax
        return x



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

num_epochs = 40
device = torch.device('cuda')

# Loads the train and test data into PyTorch tensors
training_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = transforms_train)
test_data = datasets.CIFAR10(root = "data", train = False, download = True, transform = transforms_test)

training_data, validation_set = random_split(training_data, [45000, 5000])

# Loads the data into batches
train_dataloader = DataLoader(training_data, batch_size = 500, shuffle = True)
valid_dataloader = DataLoader(validation_set, batch_size = 500, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 500, shuffle = True)

best_accuracy = 0

model = ResNet().to(device)

params = []
params += model.parameters()
# for x in model.blockset1:
#     params += x.parameters()
# for x in model.blockset2:
#     params += x.parameters()
# for x in model.blockset3:
#     params += x.parameters()

loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(params, lr = 0.001, weight_decay = 1e-6) # Best lr = 0.001
#optimizer = optim.SGD(params, lr = 0.01, momentum = 0.9, weight_decay = 1e-5)
#optimizer = optim.RMSprop(params, lr = 0.0001)

#scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 3)
scheduler = lr_s.MultiStepLR(optimizer, milestones = [32000, 48000], gamma = 0.1)

old_loss = 10000
times_worse = 0

for epoch in range(num_epochs):
    print('Epoch: ', epoch)
    train_loss = 0.0

    for batch_idx, (data, labels) in enumerate(train_dataloader):
        #print(batch_idx)
        #print(model.block3.conv2.weight)
        #print(model.blockset1[0].conv2.weight[0][0])

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

    #scheduler.step(valid_loss)
    scheduler.step()

    valid_accuracy = check_accuracy(valid_dataloader, model)
    print(valid_accuracy, '% Validation Accuracy')
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy


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

    #check_accuracy(test_dataloader, model)


print('\n')
print(check_accuracy(test_dataloader, model), '% Test Accuracy')
print('Best Validation Accuracy: ', best_accuracy)

print('Finished in %s seconds' % round(time.time() - start_time, 1))