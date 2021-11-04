"""
Author: Sam Armstrong
Date: Autumn 2021

Description: The code for generating a single sample using the model (saves the image to the local folder)
"""

import torch
from CircularNN import CircularNN
from PIL import Image
from matplotlib import cm
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import numpy as np

def run_model():
    device = torch.device('cpu')

    model = CircularNN()
    model.load_state_dict(torch.load('circular-model.pickle'))
    model.eval()

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())
    test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())
    training_data, validation_set = random_split(training_data,[50000,10000])

    # Loads the data into batches
    test_dataloader = DataLoader(test_data, batch_size = 200, shuffle = True)

    num_samples = 0
    num_correct = 0

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

            """image_array = scores[0].detach().numpy()
            image_array = image_array.reshape(28, 28)
            data = Image.fromarray(image_array)
            data = Image.fromarray(np.uint8(cm.gist_earth(image_array)*255))
            #data.show()
            data.save('Circ-Image.png')"""

            for i in range(batch_size):
                for j in range(k):
                    if -0.005 < new_labels[i][j] - scores[i][left_out_indices[i, j]] < 0.005:
                        num_correct += 1

            num_samples += batch_size * k

    print((num_correct * 100 / num_samples), '%  Correct')
    

if __name__ == '__main__':
    run_model()
