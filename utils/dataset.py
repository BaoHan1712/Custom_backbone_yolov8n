import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import random

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=True):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.img_files = [f for f in sorted(os.listdir(img_dir)) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        """Trả về số lượng ảnh trong dataset"""
        return len(self.img_files)
        
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(self.label_dir, 
                                self.img_files[idx].rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            labels = np.zeros((0, 5))
            
        # Augmentations
        if self.augment:
            img, labels = self.augment_image(img, labels)
            
        # Normalize và chuyển đổi kích thước
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1) / 255.0
            
        return torch.FloatTensor(img), torch.FloatTensor(labels)
        
    def augment_image(self, img, labels):
        # Random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img)
            if len(labels):
                labels[:, 1] = 1 - labels[:, 1]  # flip x coordinates
                
        # Random vertical flip
        if random.random() < 0.5:
            img = np.flipud(img)
            if len(labels):
                labels[:, 2] = 1 - labels[:, 2]  # flip y coordinates
                
        return img, labels

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate_fn để xử lý batch data
        Vì mỗi ảnh có số lượng objects khác nhau nên không thể stack labels
        """
        images, labels = zip(*batch)
        images = torch.stack(images)
        return images, labels
