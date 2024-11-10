import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    """
    YOLOLoss class thực hiện tính toán hàm mất mát cho YOLOv8
    Bao gồm 3 thành phần chính:
    - Box loss: Sử dụng GIoU để tính toán độ chênh lệch giữa predicted và target bounding boxes
    - Objectness loss: Xác định xác suất có object trong cell
    - Classification loss: Phân loại đối tượng
    """
    def __init__(self, num_classes=1, anchors=2):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('lambda_box', torch.tensor(1.0, device=device))
        self.register_buffer('lambda_obj', torch.tensor(1.0, device=device))
        self.register_buffer('lambda_cls', torch.tensor(1.0, device=device))
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets):
        device = predictions[0].device
        
        # Chuyển các hệ số lambda sang device
        self.lambda_box = self.lambda_box.to(device)
        self.lambda_obj = self.lambda_obj.to(device)
        self.lambda_cls = self.lambda_cls.to(device)
        
        # Khởi tạo losses là tensors
        box_loss = torch.zeros(1, device=device, requires_grad=True)
        obj_loss = torch.zeros(1, device=device, requires_grad=True)
        cls_loss = torch.zeros(1, device=device, requires_grad=True)
        num_targets = 0
        
        for pred in predictions:
            batch_size = pred.size(0)
            
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]
            pred_box = pred[..., :4]
            
            obj_mask = torch.zeros_like(pred_obj)
            
            for b in range(batch_size):
                batch_target = targets[b]
                if len(batch_target) == 0:
                    continue
                    
                grid_size = pred.size(2)
                gxy = batch_target[:, 1:3] * grid_size
                gi, gj = gxy.long().T
                
                gi = torch.clamp(gi, 0, grid_size - 1)
                gj = torch.clamp(gj, 0, grid_size - 1)
                
                for anchor_idx in range(self.anchors):
                    obj_mask[b, anchor_idx, gj, gi] = 1
                    num_targets += len(gi)
                    
                    # Box loss
                    target_box = batch_target[:, 1:5]
                    pred_anchor_box = pred_box[b, anchor_idx, gj, gi]
                    box_loss = box_loss + self.mse(pred_anchor_box, target_box).sum()
                    
                    # Class loss
                    target_cls = batch_target[:, 0].long()
                    pred_cls_anchor = pred_cls[b, anchor_idx, gj, gi]
                    cls_loss = cls_loss + self.bce(pred_cls_anchor, 
                                       F.one_hot(target_cls, 
                                       self.num_classes).float()).sum()
            
            # Objectness loss
            obj_loss = obj_loss + self.bce(pred_obj, obj_mask).sum()
        
        # Normalize losses
        num_targets = max(1, num_targets)
        box_loss = box_loss / num_targets
        obj_loss = obj_loss / num_targets
        cls_loss = cls_loss / num_targets
        
        # Tổng hợp loss với hệ số tensor
        loss = (
            self.lambda_box * box_loss +
            self.lambda_obj * obj_loss +
            self.lambda_cls * cls_loss
        )
        
        return loss, {
            'box_loss': float(box_loss.detach().cpu().numpy()),
            'obj_loss': float(obj_loss.detach().cpu().numpy()),
            'cls_loss': float(cls_loss.detach().cpu().numpy())
        }
