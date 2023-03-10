import cv2

class webcamFeed:
    def __init__(self, url, width, height):
        self.url = url
        self.width = width
        self.height = height

    def getStream(self):
        self.cap = cv2.VideoCapture(self.url)

        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

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