import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

pretrained_model = models.resnet101(pretrained=True)

class ResNet101Feats(nn.Module):
  def __init__(self):
    super(ResNet101Feats, self).__init__()
    self.submodule = pretrained_model
    self.extracted_layers = ['layer4', 'fc']

  def forward(self, x):
    output = []
    for name, module in self.submodule._modules.items():
        if name is 'fc': x = x.view(x.size(0), -1)
        x = module(x)
        if name in self.extracted_layers:
            output.append(x)        
    # print output[0].size() # (20L, 2048L, 7L, 7L)
    # print output[1].size() # (20L, 1000L)
    return output[0], output[1]
