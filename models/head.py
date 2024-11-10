import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, num_classes=1, anchors=2):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        
        # Giảm số channels trong head
        self.conv = nn.Conv2d(64, anchors * (5 + num_classes), 1)
        
    def forward(self, features):
        return [self.conv(f) for f in features]
