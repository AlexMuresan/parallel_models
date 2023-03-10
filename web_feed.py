import cv2
import numpy as np
import matplotlib.pyplot as plt

class webcamFeed:
    def __init__(self, url):
        self.url = url        

    def getStream(self):
        self.cap = cv2.VideoCapture(self.url)
        return self.cap

    def getFrame(self):
        ret, frame = self.cap.read()

        if self.cap.isOpened():
            if frame is not None:
                return frame
            else:
                print("Frame is None")
                return None
        else:
            print("Stream is not open")
            return None


if __name__ == "__main__":
    # Original link https://www.skylinewebcams.com/en/webcam/united-states/tennessee/gatlinburg/tennessee-gatlinburg.html
    # url="https://hd-auth.skylinewebcams.com/live.m3u8?a=hj1g1u62ecicbob3qfuf668la3"

    # Original link https://hdontap.com/index.php/video/stream/las-vegas-strip-live-cam
    url = "https://edge01.ny.nginx.hdontap.com/hosb5/ng_showcase-coke_bottle-street_fixed.stream/chunklist_w2119158938.m3u8"

    feed = webcamFeed(url)
    stream = feed.getStream()

    while stream.isOpened():
        frame = feed.getFrame()

        if frame is not None:
            cv2.imshow("frame", frame)

            if cv2.waitKey(22) & 0xFF == ord('q'):
                print("Done")
                break