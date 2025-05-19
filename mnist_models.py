import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.gelu = nn.GELU()
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SmallVGG(nn.Module):
    def __init__(self, in_chans=1, num_classes=10, input_size=28):
        super(SmallVGG, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self._initialize_classifier()

    def _initialize_classifier(self):
        with torch.no_grad():
            temp_input = torch.zeros(1, self.in_chans, self.input_size, self.input_size)
            temp_output = self.features(temp_input)
            flattened_size = temp_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.GELU(),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class LogisticRegressionModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
    
    
from collections import OrderedDict
from torch import nn


HASH = '5c427c9c'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


def CORnet_Z():
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model

def CORnet_Z_cifar(inchans=3):
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(inchans, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 10)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model