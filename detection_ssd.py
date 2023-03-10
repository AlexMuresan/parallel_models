import cv2
import torch

from webcamFeed import webcamFeed

if __name__ == "__main__":
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    # Original link https://hdontap.com/index.php/video/stream/las-vegas-strip-live-cam
    url = "https://edge01.ny.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w2119158938.m3u8"

    feed = webcamFeed(url, 224, 224)
    stream = feed.getStream()

    ssd_model.eval()

    with torch.no_grad():
        while stream.isOpened():
            frame = feed.getFrame()

            if frame is not None:
                image = frame
