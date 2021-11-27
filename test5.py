import numpy as np
import torch

a = torch.transpose(torch.tensor([1,2,3]).unsqueeze(1),0,1)
b = torch.tensor([1,2,3]).unsqueeze(1)
print(a.shape)
print(b.shape)
print(a.mm(b).shape)
