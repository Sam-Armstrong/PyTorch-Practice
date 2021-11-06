"""
Implementation of the data-efficient image transformer architecture created by the Facebook AI team.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import time
import torch.optim.lr_scheduler as lr_s

class DeIT(nn.Module):

    def __init__(self):
        super(DeIT, self).__init__()

    def forward(self, x):
        return x