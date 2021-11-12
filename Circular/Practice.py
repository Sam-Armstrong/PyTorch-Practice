import torch.nn as nn
import torch.nn.functional as F
import torch

t = torch.rand(2, 3, 3, 3)
print(t)

out = F.pad(t, (0, 0, 0, 0, 0, 2))
print(out)