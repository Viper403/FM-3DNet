import torch
import torch.nn as nn
import torch.nn.functional as F

model_path = '../exp/models/200.pth'
checkpoint = torch.load(model_path)
print(checkpoint['state_dict'].keys())