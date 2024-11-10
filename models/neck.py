import torch
import torch.nn as nn

class LightPAN(nn.Module):
    def __init__(self, in_channels=[64, 128, 256], out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, 1)
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, features):
        [x1, x2, x3] = features
        
        p3 = self.conv3(x3)
        p2 = self.conv2(x2) + self.upsample(p3)
        p1 = self.conv1(x1) + self.upsample(p2)
        
        return [p1, p2, p3]
