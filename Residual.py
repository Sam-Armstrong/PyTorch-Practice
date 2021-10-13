import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, dim, kernel_size):
        super(Residual, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, groups = dim, padding = "same")
        self.gelu = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        old_x = x
        x = self.conv(x)
        x = self.gelu(x)
        x = self.batch_norm(x)
        return x + old_x