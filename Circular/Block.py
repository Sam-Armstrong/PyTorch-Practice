import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(1, stride=stride)

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, stride = 1, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    
    def forward(self, x):
        res = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Applies the optional downsampling
        if self.stride > 1:
            res = F.pad(res, (0, 0, 0, 0, 0, self.out_channels - self.in_channels))
            res = self.pooling(res)

        x += res
        x = self.relu(x)

        return x
