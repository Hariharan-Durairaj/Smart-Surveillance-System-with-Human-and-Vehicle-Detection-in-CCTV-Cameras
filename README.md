# Smart Surveillance System with Human and Vehicle Detection in CCTV Cameras

## Project Overview

This project uses **YOLOv5** to perform real-time human and vehicle detection from an **RTSP stream** captured via an IP camera. The video stream is processed to detect humans and vehicles, and based on detection, the system saves the video at different frame rates. The key features are:

- **Low FPS Recording (1 FPS)** when there are **no detections** to save disk space.
- **High FPS Recording (25 FPS)** when a **detection is found** (i.e., human or vehicle).
- **Separate video files** are saved for detection and no-detection states, each with timestamps for easy tracking.

### Key Features:
- Detects **human** and **vehicle** (car, motorcycle, bus, truck) using YOLOv5.
- Saves video in **separate files** based on detection.
  - **No detection**: Saved at **1 FPS** to save storage.
  - **Detection**: Saved at **25 FPS** for clear visibility.
- Supports **real-time video display** with detection boxes overlayed on the frames.
- Outputs video files with **timestamps** and a specific naming convention indicating detection.

## Screenshot Samples

Here are some sample screenshots demonstrating the output:

- **Detection in Video**:  

![alt text](images\\Detection.png)
  

- **Saved Video Files**:  

![alt text](images\\FIle_naming.png)
 

## Installation Instructions

You can set up the environment using **conda** or **pip**. Below are the steps to install the necessary dependencies.

### Using Conda:
1. Create a new conda environment:
   ```bash
   conda create --name detection-env python=3.8
   ```

2. Activate the environment:
   ```bash
   conda activate detection-env
   ```

3. Install required dependencies:
   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   conda install -c conda-forge opencv
   conda install numpy
   conda install pandas
   conda install ultralytics
   ```

### Using Pip:

1. Install dependencies using pip:
   ```bash
   pip install torch torchvision torchaudio
   pip install opencv-python
   pip install numpy
   pip install pandas
   pip install ultralytics
   ```

2. Download YOLOv5 model weights (this is done automatically by `torch.hub.load` in the code).

### Using ```requirements.txt```:

1. Go to the folder:
   ```bash
   cd Smart-Surveillance-System-with-Human-and-Vehicle-Detection-in-CCTV-Cameras
   ```

2. Install dependencies using ```requirements.txt```:
   ```bash
   pip install -r requirements.txt
   ```

## How the Code Works

### Video Stream Processing
1. **RTSP Stream**:  
   The script connects to an **IP camera** using the RTSP stream URL and grabs frames for processing.

2. **Detection Loop**:  
   Every frame is passed through the **YOLOv5 model** to detect objects like humans, cars, motorcycles, buses, and trucks.
   - If a detection is found, the video is saved at **25 FPS** with detection boxes overlayed on the video.
   - If no detection is found, the video is saved at **1 FPS** to save storage space.

3. **Saving Video**:  
   Video frames are saved into **separate files**:
   - Files with **"Alert!!!"** in the name indicate that detections were made (i.e., humans or vehicles).
   - Files without **"Alert!!!"** are saved at **1 FPS** (i.e., when there are no detections).

4. **Grace Period**:  
   After detection, the system will continue saving at high FPS for a **grace period of 30 seconds** (to capture events after detection), then revert to low FPS if no further detection occurs.

### Video File Naming Convention:
- Files with detection will have names like:
  ```
  recording_2025-04-24_14-20-30_Alert!!!.mp4
  ```
- Files with no detection will have names like:
  ```
  recording_2025-04-24_14-20-30.mp4
  ```

### Real-Time Detection and Display
- The video is displayed in real-time with detection boxes drawn on the frames:
  - **Green boxes** indicate detected humans or vehicles.
  - Detection labels (e.g., "person", "car") are shown next to the bounding boxes.ppytho

## How to Run the Code

1. **Clone the repository** (if applicable):
   ```bash
   git clone https://github.com/Hariharan-Durairaj/Smart-Surveillance-System-with-Human-and-Vehicle-Detection-in-CCTV-Cameras.git
   ```

2. **Edit the ```smart_surveillance.py```**:
   ```bash
   # RTSP stream URL and credentials
   # Replace 'username', 'password', and 'ip_address' with your actual RTSP stream credentials
   RTSP_URL = 'rtsp://username:password@ip_address/stream'
   ```
    If you don't know the url for the stream download [Onvif Device Manager](https://sourceforge.net/projects/onvifdm/files/latest/download). In this software there is an option to enter the username and password for the IP camera. Which can then be used to see the live view of the camera including the stream URL. Sometime you might need to manually change the URL, to add the username and password like this:  ```rtsp://username:password@url_from_onvif_device_manager```.  

3. Monitor the **real-time video display** and check the **recordings** in the specified directory (`recordings/`).

## File Saving Mechanism

The videos are saved as `.mp4` files in the `recordings/` directory. The file names will include the timestamp of the recording, and videos with detections will include "Alert!!!" in the name.

### Example:
- `ALERT!!!_record_20250424_183242_15fps.mp4` (contains detection)
- `record_20250424_183142_1fps.mp4` (no detection)

The code saves recordings in chunks of **1 hour** (or the specified `CHUNK_DURATION`) to avoid creating excessively large files.

## License

This project is licensed under the MIT License.

---

## Troubleshooting

- **RTSP stream errors**: Ensure the IP camera RTSP URL is correct and accessible from your machine.
- **Detection issues**: If detection is too slow or inaccurate, consider switching to a larger model (`yolov5m` or `yolov5l`) or fine-tuning on your specific dataset.
