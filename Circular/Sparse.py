import torch
import torch.nn as nn
import math

class Sparse(nn.Module):

    def __init__(self, input_size, output_size, connections_per_neuron = 2):
        super(Sparse, self).__init__()
        self.connections = connections_per_neuron
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size)
        self.sparse_layer = []
        
        self.individual_input_size = int(math.ceil(input_size / output_size))
        #print('Individual input size: ', self.individual_input_size)

        for i in range(input_size):
            self.sparse_layer.append(nn.Linear(self.individual_input_size, connections_per_neuron))


    def forward(self, x):
        #print(self.sparse_layer[294].weight)
        x = self.sparse_forward(x)
        return x

    def sparse_forward(self, x):
        x = x.reshape(x.shape[0], 784)
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
        
        return y
