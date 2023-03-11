import cv2
import torch
import numpy as np
from torchvision import transforms

from effdet.backbone import EfficientDetBackbone
from effdet.efficientdet.utils import BBoxTransform, ClipBoxes
from effdet.utils.utils import (invert_affine, postprocess, preprocess,
                                preprocess_video)
from webcamFeed import webcamFeed


# function for display
def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        
        return imgs[i]


if __name__ == "__main__":

    ## Effdet Stuff
    compound_coef = 0
    force_input_size = None  # set None to use default size

    threshold = 0.2
    iou_threshold = 0.2

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

    # load model
    model_effdet = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model_effdet.load_state_dict(torch.load(f'efficientdet-d{compound_coef}.pth'))
    model_effdet.requires_grad_(False)
    model_effdet.eval()

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    
    
    ## YOLO Stuff
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    ## Streaming stuff
    # Original link https://hdontap.com/index.php/video/stream/las-vegas-strip-live-cam
    url = "https://edge01.ny.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w2119158938.m3u8"

    feed = webcamFeed(url, 224, 224)
    stream = feed.getStream()
    frame_idx = 0

    while stream.isOpened():
        frame = feed.getFrame()

        if frame is not None:
            image = frame

            scale_percent = 50 # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

            ori_imgs, framed_imgs, framed_metas = preprocess_video(image, max_size=input_size)
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
            x = x.to(torch.float32).permute(0, 3, 1, 2)

            img_yolo = image.copy()

            with torch.no_grad():
                features, regression, classification, anchors = model_effdet(x)

                out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
                
                output_yolo = model_yolo(image)
                
            out_effdet = invert_affine(framed_metas, out)
            img_effdet= display(out, ori_imgs)

            for detection in output_yolo.xyxy[0]:
                xmin = int(detection[0])
                ymin = int(detection[1])

                xmax = int(detection[2])
                ymax = int(detection[3])

                cv2.rectangle(img_yolo, (xmin, ymin), (xmax, ymax), (0,255,0), 1)

            # cv2.imshow("YOLOV5", img_yolo)
            # cv2.imshow("EfficientNet", img_effdet)

            img_combined = np.concatenate((img_yolo, img_effdet), axis=1)
            cv2.imshow("YoloV5 vs EfficientDet", img_combined)

            if cv2.waitKey(22) & 0xFF == ord('q'):
                print("Done")
                break




