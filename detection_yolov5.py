import cv2
import torch

from torchvision import transforms

from webcamFeed import webcamFeed


if __name__ == "__main__":
    # Original link https://hdontap.com/index.php/video/stream/las-vegas-strip-live-cam
    url = "https://edge01.ny.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w2119158938.m3u8"

    feed = webcamFeed(url, 224, 224)
    stream = feed.getStream()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    with torch.no_grad():
        while stream.isOpened():
            frame = feed.getFrame()

            if frame is not None:
                image = frame

                scale_percent = 50 # percent of original size
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                dim = (width, height)

                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                output = model(image)

                for detection in output.xyxy[0]:
                    xmin = int(detection[0])
                    ymin = int(detection[1])

                    xmax = int(detection[2])
                    ymax = int(detection[3])

                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 1)

                cv2.imshow("Live Stream", image)

                if cv2.waitKey(22) & 0xFF == ord('q'):
                    print("Done")
                    break

