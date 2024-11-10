import cv2
import torch
import numpy as np
import yaml
from models.yolo import YOLOv8

class YOLOPredictor:
    def __init__(self):
        # Load config
        config_path = 'configs/model_config.yaml'
        checkpoint_path = 'checkpoints/best.pth'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = YOLOv8(
            num_classes=self.config['model']['head']['num_classes'],
            anchors=self.config['model']['head']['anchors']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.confidence = 0.25
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    @torch.cuda.amp.autocast()
    def detect_image(self, frame):
        # Tiền xử lý frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_size = 640
        orig_h, orig_w = frame.shape[:2]
        
        # Resize và normalize
        img = cv2.resize(img, (img_size, img_size))
        img = img.transpose(2, 0, 1) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img)[0]
            
        predictions = predictions[0]
        
        # Lọc predictions theo confidence
        mask = predictions[..., 4] > self.confidence
        if torch.any(mask):
            filtered_preds = predictions[mask]
            
            for pred in filtered_preds:
                # Lấy tọa độ box và thông tin
                x1, y1, x2, y2 = pred[:4].cpu().numpy()
                conf = float(pred[4])
                cls_id = int(pred[5])
                
                # Scale về kích thước gốc
                x1 = int(x1 * orig_w / img_size)
                y1 = int(y1 * orig_h / img_size)
                x2 = int(x2 * orig_w / img_size)
                y2 = int(y2 * orig_h / img_size)
                
                # Vẽ bbox với màu xanh lá
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ label phía trên bbox
                label = f'Benh: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        return frame

def main():
    predictor = YOLOPredictor()
    
    # Đường dẫn tới ảnh
    image_path = 'cay.JPG'
    
    # Đọc ảnh từ file
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (640, 480))
    
    if frame is None:
        print("Không thể đọc ảnh.")
        return
    
    # Detect và vẽ kết quả
    frame = predictor.detect_image(frame)
    
    # Hiển thị ảnh kết quả
    cv2.imshow('YOLO Detection', frame)
    cv2.waitKey(0)  # Nhấn bất kỳ phím nào để đóng cửa sổ
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
