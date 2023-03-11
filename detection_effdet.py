import cv2
import torch

from effdet.backbone import EfficientDetBackbone
from effdet.efficientdet.utils import BBoxTransform, ClipBoxes
from effdet.utils.utils import preprocess, invert_affine, postprocess, preprocess_video

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

    use_float16 = False

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
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(f'efficientdet-d{compound_coef}.pth'))
    model.requires_grad_(False)
    model.eval()

    if use_float16:
        model = model.half()

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()


    # Original link https://hdontap.com/index.php/video/stream/las-vegas-strip-live-cam
    url = "https://edge01.ny.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w2119158938.m3u8"

    feed = webcamFeed(url, 224, 224)
    stream = feed.getStream()

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
            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            # model predict
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
                
            # result
            out = invert_affine(framed_metas, out)
            img_show = display(out, ori_imgs)

            # show frame by frame
            # cv2.imshow('frame',img_show)

            cv2.imshow("Live Stream", img_show)

            if cv2.waitKey(22) & 0xFF == ord('q'):
                print("Done")
                break
