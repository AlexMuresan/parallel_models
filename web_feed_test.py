import cv2
import numpy as np
import matplotlib.pyplot as plt


url = "https://hd-auth.skylinewebcams.com/live.m3u8?a=hj1g1u62ecicbob3qfuf668la3"
feed = cv2.VideoCapture(url)

while feed.isOpened():
    ret, frame = feed.read()

    if ret and (frame is not None):
        cv2.imshow("frame", frame)

        if cv2.waitKey(22) & 0xFF == ord('q'):
            break  
    else:
        print("Reading stream failed")