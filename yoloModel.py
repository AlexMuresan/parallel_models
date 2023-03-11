import cv2
import torch

from detectionModel import detectionModel

class yoloModel(detectionModel):
    def __init__(self, model_name="YoloV5", pretrained=True):
        super().__init__(model_name=model_name)
        
        self.pretrained = pretrained

    def load_model(self):
        print(f"Loading {self.model_name}...")
        torch.hub.set_dir('./weights/ultralytics_yolov5_master')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)

        print(f"Model {self.model_name} loaded.")
        return self.model
    
    def infer(self, image):
        with torch.no_grad():
            self.preds = self.model(image)
    
    def draw_detections(self, image, threshold=0.3):
        for detection in self.preds.xyxy[0]:
            confidence = detection[4]
            if confidence >= threshold:
                x1 = int(detection[0])
                y1 = int(detection[1])

                x2 = int(detection[2])
                y2 = int(detection[3])

                object_name = self.obj_list[detection[5].numpy().astype(int)]

                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 1)
                cv2.putText(image, '{}, {:.3f}'.format(object_name, confidence),
                                (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
                
        return image