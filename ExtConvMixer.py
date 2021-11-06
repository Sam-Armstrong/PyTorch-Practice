import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import time
import torch.optim.lr_scheduler as lr_s

class ExtConvMixer(nn.Module):
    def __init__(self):
        super(ExtConvMixer, self).__init__()

        # Model parameters
        dim = 28
        patch_size = 2
        kernel_size = 3
        n_classes = 10

        self.conv1 = nn.Conv2d(1, dim, kernel_size = patch_size, stride = patch_size) # Patch encoding
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, groups = dim, padding = "same") # Depthwise convolution
        self.conv3 = nn.Conv2d(dim, dim, kernel_size = 1) # Pointwise convolution
        self.conv4 = nn.Conv2d(dim, dim, kernel_size = patch_size, stride = patch_size) # Patch encoding
        self.conv5 = nn.Conv2d(dim, dim, kernel_size, groups = dim, padding = "same") # Depthwise convolution
        self.conv6 = nn.Conv2d(dim, dim, kernel_size = 1) # Pointwise convolution
        
        self.fc = nn.Linear(dim, n_classes)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.bn4 = nn.BatchNorm2d(dim)
        self.bn5 = nn.BatchNorm2d(dim)
        self.bn6 = nn.BatchNorm2d(dim)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Patch encoding
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.bn1(x)

        # Residual
        y = x.detach().clone()
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.bn2(x)
        x += y

        # Pointwise convolution
        x = self.conv3(x)
        x = self.gelu(x)
        x = self.bn3(x)

        """# Patch encoding
        x = self.conv4(x)
        x = self.gelu(x)
        x = self.bn4(x)

        # Residual
        y = x.detach().clone()
        x = self.conv5(x)
        x = self.gelu(x)
        x = self.bn5(x)
        x += y

        # Pointwise convolution
        x = self.conv6(x)
        x = self.gelu(x)
        x = self.bn6(x)"""

        # Average pool and linear for classification
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x



start_time = time.time()

num_epochs = 20
device = torch.device('cpu')

# Loads the train and test data into PyTorch tensors
training_data = datasets.MNIST(root = "data", train = True, download = True, transform = ToTensor())
test_data = datasets.MNIST(root = "data", train = False, download = True, transform = ToTensor())
training_data, validation_set = random_split(training_data,[50000,10000])

# Loads the data into batches 
train_dataloader = DataLoader(training_data, batch_size = 200, shuffle = True)
valid_dataloader = DataLoader(validation_set, batch_size = 200, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 200, shuffle = True)

model = ExtConvMixer().to(device)

loss_f = nn.CrossEntropyLoss() #nn.MSELoss()#nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0)
scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 2)

old_loss = 10000
times_worse = 0

for epoch in range(num_epochs):
    print(epoch)
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
    print(valid_loss)

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