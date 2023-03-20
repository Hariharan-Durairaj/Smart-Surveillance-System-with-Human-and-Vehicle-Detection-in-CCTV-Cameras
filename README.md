# Enhanced Surveillance Systems with Human and Object Detection in IP Cameras

This script uses the YOLOv5m model to detect objects in a low-resolution video stream obtained via RTSP protocol. If the model detects a person or car, it sets a flag to start recording a high-resolution video stream using RTSP of the high-resolution profile. Then recorded videos are saved in a directory called "motion_detection".

This code will save a lot of disk space, increase the longevity of the hard disk, and decrease the number of false alarms compared to a normal camera using motion detection. 

## Prerequisites
Python 3.7 or higher
OpenCV, PyTorch, and keyboard libraries
Access credentials (username and password) for the RTSP cameras
RTSP cameras with URLs for low and high-resolution video streams. To find the RTSP address of the camera domnload [Onvif Device Manager](https://onvif-device-manager.software.informer.com/download/?ca1a3192). 
Keep Low resolution profile in 1 FPS to reduce GPU and CPU usage

## Installation
To install the required libraries, run the following command in your terminal:
'''
pip install -r requirements.txt
'''

To install pytorch gpu,run the following command in your terminal:

For conda
'''
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
'''

For pip
'''
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
'''

For more information about [pytorch](https://pytorch.org/).

You can clone this repository to obtain the script and sample RTSP URLs.




## Usage

Open the nodebook through any application like jupiter, vs code

The script will open two threads - one for detecting objects in low resolution and the other for recording high-resolution video streams. You can change the RTSP URLs, display flag, recording timeout, and output directory in the script's global variables.

While running the script, you can press the "q" key to stop the program and save any ongoing recordings.

## Acknowledgments
This script uses the YOLOv5m model from the Ultralytics repository and the OpenCV library for video processing.