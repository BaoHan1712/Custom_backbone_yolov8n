import torch
import torch.nn as nn
from .backbone import LightCSPDarknet

class YOLOv8(nn.Module):
    def __init__(self, num_classes=1, anchors=2):
        super().__init__()
        self.backbone = LightCSPDarknet()
        
        # Tính toán số channels output
        self.out_channels = anchors * (5 + num_classes)  
        
        # Neck layer với số channels phù hợp
        self.neck = nn.Sequential(
            nn.Conv2d(128, 128, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, self.out_channels, 1)
        )
        
        self.num_classes = num_classes
        self.anchors = anchors
        
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Lấy feature map cuối cùng từ backbone
        out = features[-1]  # [batch_size, 128, H, W]
        
        # Process through neck
        out = self.neck(out)  # [batch_size, anchors*(5+num_classes), H, W]
        
        # Reshape output
        batch_size, _, height, width = out.shape
        out = out.view(batch_size, self.anchors, 5 + self.num_classes, height, width)
        out = out.permute(0, 1, 3, 4, 2)  # [batch_size, anchors, H, W, 5+num_classes]
        
        return [out]
