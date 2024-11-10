import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models.yolo import YOLOv8
from utils.dataset import YOLODataset
from utils.loss import YOLOLoss
import yaml
import os
from tqdm import tqdm
import gc

def get_scheduler(optimizer, config, train_loader):
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'],
            epochs=config['training']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['min_lr']
        )
    elif scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    return scheduler

def train():
    # Tạo thư mục checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Thiết lập device và scaler cho mixed precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()
    print(f"Using device: {device}")

    # Load config
    with open('configs/model_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Thiết lập CUDA
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Khởi tạo model
    model = YOLOv8(
        num_classes=config['model']['head']['num_classes'],
        anchors=config['model']['head']['anchors']
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Khởi tạo datasets
    train_dataset = YOLODataset(
        img_dir='data/train/images',
        label_dir='data/train/labels',
        img_size=640,
        augment=True
    )
    
    val_dataset = YOLODataset(
        img_dir='data/val/images', 
        label_dir='data/val/labels',
        img_size=640,
        augment=False
    )
    
    # DataLoader với pin_memory=True cho GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn
    )
    
    # Optimizer với amp
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = get_scheduler(optimizer, config, train_loader)
    
    criterion = YOLOLoss(
        num_classes=config['model']['head']['num_classes'],
        anchors=config['model']['head']['anchors']
    ).to(device)

    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
            images = images.to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in targets]
            
            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss, loss_items = criterion(outputs, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            
            if config['training']['scheduler']['type'] == "OneCycleLR":
                scheduler.step()
                
            epoch_loss += loss.item()
            
            # Dọn memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Validation với torch.cuda.amp.autocast()
        model.eval()
        val_loss = 0
        
        with torch.no_grad(), autocast():
            for images, targets in tqdm(val_loader):
                images = images.to(device, non_blocking=True)
                targets = [t.to(device, non_blocking=True) for t in targets]
                
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss = epoch_loss / len(train_loader)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'config': config
            }
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print(f'Saved best model with val_loss: {best_val_loss:.4f}')
        
        # Dọn memory sau mỗi epoch
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    train()
