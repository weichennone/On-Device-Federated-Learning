#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class CNNMNIST(nn.Module):
    """7,106"""
    def __init__(self, args):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, 3, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 8, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 8)
        x_out = F.relu(self.fc1(x))
        x = self.fc2(x_out)
        return x, x_out

