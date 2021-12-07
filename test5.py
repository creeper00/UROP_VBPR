import numpy as np
import torch

model = torch.load("../datasets/model2.pt")
torch.save(model.state_dict(), "../datasets/model_s.pt")
