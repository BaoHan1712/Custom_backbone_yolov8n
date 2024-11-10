import cv2
import torch
import numpy as np
import yaml
from models.yolo import YOLOv8

class YOLOPredictor:
    def __init__(self):
        # Load config
        config_path = 'configs/model_config.yaml'
        checkpoint_path = 'checkpoints/best_model.pth'
        
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
        self.colors = [(0,255,0), (255,0,0), (0,0,255)]

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
                # Lấy tọa độ box
                x1, y1, x2, y2 = pred[:4].cpu().numpy()
                
                # Scale về kích thước gốc
                x1 = int(x1 * orig_w / img_size)
                y1 = int(y1 * orig_h / img_size)
                x2 = int(x2 * orig_w / img_size)
                y2 = int(y2 * orig_h / img_size)
                
                conf = float(pred[4])
                cls_id = int(pred[5])
                
                # Vẽ box và label
                color = self.colors[cls_id % len(self.colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f'Class {cls_id}: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        return frame

def main():
    predictor = YOLOPredictor()
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    # Thiết lập cửa sổ
    cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect và vẽ kết quả
        frame = predictor.detect_image(frame)
        
        # Hiển thị hướng dẫn
        cv2.putText(frame, 'Press Q to quit', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị frame
        cv2.imshow('YOLO Detection', frame)
        
        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
