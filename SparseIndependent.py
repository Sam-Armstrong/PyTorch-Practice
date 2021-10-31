import torch
import torch.nn as nn
import math
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import time
import torch.optim.lr_scheduler as lr_s


class Sparse(nn.Module):

    def __init__(self, input_size = 784, output_size = 784, connections_per_neuron = 2):
        super(Sparse, self).__init__()
        self.connections = connections_per_neuron
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size)
        self.sparse_layer = []
        self.fc = nn.Linear(784, 10)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim = -1)
        
        self.individual_input_size = int(math.ceil(input_size / output_size))
        print('Individual input size: ', self.individual_input_size)

        for i in range(input_size):
            self.sparse_layer.append(nn.Linear(self.individual_input_size, connections_per_neuron))


    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        
        y = torch.zeros((x.shape[0], self.output_size))

        for i in range(self.input_size):
            if i+self.connections >= 784:
                cutoff = i + self.connections - 784
                if cutoff > 0:
                    y[:, 784 - cutoff:784] += self.sparse_layer[i](x[:, i:i+1])[:, :cutoff] # End of array
                if self.connections - cutoff > 0:
                    y[:, :self.connections - cutoff] += self.sparse_layer[i](x[:, i:i+1])[:, cutoff:] # Start of array

            else:
                y[:, i:i+self.connections] += self.sparse_layer[i](x[:, i:i+1])
        
        x = y
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def sparse_forward(self, x):
        y = torch.zeros((x.shape[0], self.output_size))

        for i in range(self.input_size):
            for n in range(self.connections):
                ## Minor bug here - when I is between 784 and 784-connections, the output nees to be added to the ends of the array
                if i >= 784 - self.connections:
                    j = i - (784 - self.connections)
                else:
                    j = i

                y[:, j+n:j+n+2] += self.sparse_layer[i](x[:, j+n:j+n+1])
        return y





start_time = time.time()

num_epochs = 1
device = torch.device('cpu')

# Loads the train and test data into PyTorch tensors
training_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())
test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())
training_data, validation_set = random_split(training_data,[50000,10000])

# Loads the data into batches 
train_dataloader = DataLoader(training_data, batch_size = 100, shuffle = True)
valid_dataloader = DataLoader(validation_set, batch_size = 100, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 100, shuffle = True)

model = Sparse().to(device)

#params = list(model.parameters()) + [x.parameters() for x in model.sparse_layer]

params = []
params += model.parameters()
for x in model.sparse_layer:
    params += x.parameters()

loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(params, lr = 0.01, weight_decay = 0)
scheduler = lr_s.ReduceLROnPlateau(optimizer, 'min', patience = 2)

old_loss = 10000
times_worse = 0

for epoch in range(num_epochs):
    train_loss = 0.0

    for batch_idx, (data, labels) in enumerate(train_dataloader):
        print(batch_idx)
        #print(model.sparse_layer[295].weight)
        data = data.to(device = device)
        labels = labels.to(device = device)

        #data = data.reshape(data.shape[0], -1) # Flattens the data into a long vector

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