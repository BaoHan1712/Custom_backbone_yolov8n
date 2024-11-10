import torch
import torch.nn as nn

class LightCSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        # Số channels và layers
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)  
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.act = nn.ReLU()
        
    def forward(self, x):
        x1 = self.act(self.bn1(self.conv1(x)))
        x2 = self.act(self.bn2(self.conv2(x1)))
        x3 = self.act(self.bn3(self.conv3(x2)))
        
        return [x2, x3]  # Return feature maps với channels tương ứng [64, 128]
