import cv2
import torch

from effdet.backbone import EfficientDetBackbone
from effdet.efficientdet.utils import BBoxTransform, ClipBoxes
from effdet.utils.utils import (invert_affine, postprocess, preprocess,
                                preprocess_video)

from detectionModel import detectionModel


class efficientDetModel(detectionModel):
    def __init__(self, model_name="EfficientNet"):
        super().__init__(model_name=model_name)
        self.compound_coef = 0

        self.threshold = 0.2
        self.iou_threshold = 0.2

        # tf bilinear interpolation is different from any other's, just make do
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = self.input_sizes[self.compound_coef]

        # boxes
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        
    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, 
                                          num_classes=len(self.obj_list))
        self.model.load_state_dict(torch.load(f'efficientdet-d{self.compound_coef}.pth'))
        self.model.requires_grad_(False)
        self.model.eval()

        print(f"Model {self.model_name} loaded.")

        return self.model

    def infer(self, image):
        ori_imgs, framed_imgs, framed_metas = preprocess_video(image, max_size=self.input_size)
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            out = postprocess(x,anchors, regression, classification, self.regressBoxes, 
                              self.clipBoxes, self.threshold, self.iou_threshold)
            
        self.preds = invert_affine(framed_metas, out)

    def draw_detections(self, image, threshold=0.3):
        for j in range(len(self.preds[0]['rois'])):
            score = float(self.preds[0]['scores'][j])
            if score >= threshold:
                (x1, y1, x2, y2) = self.preds[0]['rois'][j].astype(int)
                obj = self.obj_list[self.preds[0]['class_ids'][j]]

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(image, '{}, {:.3f}'.format(obj, score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)
            
        return image