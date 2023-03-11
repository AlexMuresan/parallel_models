import cv2
import numpy as np

from yoloModel import yoloModel
from webcamFeed import webcamFeed
from efficientDetModel import efficientDetModel


if __name__ == "__main__":
    ## Streaming stuff
    # Original link https://hdontap.com/index.php/video/stream/las-vegas-strip-live-cam
    url = "https://edge01.ny.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w2119158938.m3u8"

    feed = webcamFeed(url, 224, 224)
    stream = feed.getStream()
    frame_idx = 0

    yolo = yoloModel(pretrained=True)
    yolo.load_model()

    effdet = efficientDetModel()
    effdet.load_model()

    while stream.isOpened():
        frame = feed.getFrame()

        if frame is not None:
            image = frame

            scale_percent = 50 # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

            yolo.infer(image)
            img_yolo = yolo.draw_detections(image.copy())
            # cv2.imshow("YOLOV5", img_yolo)

            effdet.infer(image)
            img_effdet = effdet.draw_detections(image.copy())
            # cv2.imshow("EfficientNet", img_effdet)

            img_combined = np.concatenate((img_yolo, img_effdet), axis=1)
            cv2.imshow("YoloV5 vs EfficientDet", img_combined)

            if cv2.waitKey(22) & 0xFF == ord('q'):
                print("Done")
                break




