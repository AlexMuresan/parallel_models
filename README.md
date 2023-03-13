# Parallel models

## Running the script

1. Install requirements

   - `pip install -r requirements.txt`

2. Run detection script

   - `python detection_parallel.py`

## Implementation details

The script uses OpenCV to read from a live videostream in Las Vegas then, using PyTorch, it loads 2 object detection models and displays the results side by side in the same window.

The two models I used:

1. YoloV5 from [Ultralytics](https://github.com/ultralytics/yolov5)
2. EfficientDet from [Yet Another EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
