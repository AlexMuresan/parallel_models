import cv2
import torch

from effdet.backbone import EfficientDetBackbone
from webcamFeed import webcamFeed

if __name__ == "__main__":

    compound_coef = 0
    force_input_size = None  # set None to use default size

    threshold = 0.2
    iou_threshold = 0.2


    # Original link https://hdontap.com/index.php/video/stream/las-vegas-strip-live-cam
    url = "https://edge01.ny.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w2119158938.m3u8"

    feed = webcamFeed(url, 224, 224)
    stream = feed.getStream()

    with torch.no_grad():
        while stream.isOpened():
            frame = feed.getFrame()

            if frame is not None:
                cv2.imshow("Live Stream", frame)

                if cv2.waitKey(22) & 0xFF == ord('q'):
                    print("Done")
                    break
