"""
nn_models.py: contains the different models used for image classification

This module contains different Neural Networks in form of Classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.50)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.55)
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y = F.log_softmax(x, dim=1)
        return y


class MobileNetV3SmallCNN(nn.Module):
    def __init__(self, req_grad=False, num_classes=10):
        super().__init__()
        self.mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)

        # Freeze all layers except the classifier (optional)
        for param in self.mobilenet_v3_small.parameters():
            param.requires_grad = req_grad

        # Replace the classifier with a new one for our task
        self.mobilenet_v3_small.classifier[3] = nn.Linear(
            in_features=1024, out_features=num_classes
        )

    def forward(self, x):
        return self.mobilenet_v3_small(x)
