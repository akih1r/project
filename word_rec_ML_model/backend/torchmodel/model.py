import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim=(1, 128, 32), output_size=36):
        super(CNN, self).__init__()
        
        
        # input: (1, 128, 32) -> output: (30, 128, 32) -> pool: (30, 64, 16)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(30)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        # input: (30, 64, 16) -> output: (64, 64, 16) -> pool: (64, 32, 8)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        # 64(fn2) * 32(h) * 8(w) = 16384
        self.affine1 = nn.Linear(64 * 32 * 8, 100) 
        self.bn3 = nn.BatchNorm1d(100)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.affine2 = nn.Linear(100, output_size)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.relu3(self.bn3(self.affine1(x)))
        x = self.dropout1(x)
        x = self.affine2(x)
        
        return x 