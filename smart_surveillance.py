import cv2
import time
import torch
import numpy as np
from datetime import datetime, timedelta
import os
import threading
from queue import Queue

# === CONFIGURATION ===
# RTSP stream URL and credentials
# Replace 'username', 'password', and 'ip_address' with your actual RTSP stream credentials
RTSP_URL = 'rtsp://username:password@ip_address/stream'

# Frames per second when there are no detections (to save space)
LOW_FPS = 1

# Frames per second when a person or vehicle is detected (higher quality capture)
HIGH_FPS = 15

# Time in seconds to wait before switching back to LOW_FPS after last detection
NO_DETECTION_GRACE_PERIOD = 30  # seconds

# Duration in seconds for each saved video file (default is 1 hour)
CHUNK_DURATION = 3600  # seconds (1 hour)

# Directory where the recorded video chunks will be saved
OUTPUT_DIR = 'recordings'

# Flag to enable or disable video display on screen for monitoring
DISPLAY_VIDEO = False

# Dimensions in which the video frames are resized
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080

# Change to 'yolov5n' if you have very low specs or 'yolov5m' for medium-sized models if you have a higher CPU spec
YOLO_MODEL = 'yolov5s'

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL, pretrained=True)
model.classes = [0, 2, 3, 5, 7]  # Person, Car, Motorcycle, Bus, Truck
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def create_video_writer(fps):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if fps==1:
        filename = os.path.join(OUTPUT_DIR, f"record_{timestamp}_{fps}fps.mp4")
    else:
        filename = os.path.join(OUTPUT_DIR, f"ALERT!!!_record_{timestamp}_{fps}fps.mp4")
    return cv2.VideoWriter(filename, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

# === THREAD 1: FRAME CAPTURE ===
class FrameGrabber(threading.Thread):
    def __init__(self, cap, frame_queue):
        super().__init__()
        self.cap = cap
        self.queue = frame_queue
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                if not self.queue.full():
                    self.queue.put(frame)
            else:
                print("Failed to read from stream")
                time.sleep(1)

    def stop(self):
        self.running = False

# === THREAD 2: DETECTION ===
class Detector(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.queue = frame_queue
        self.latest_detection = False
        self.latest_frame = None
        self.running = True

    def run(self):
        while self.running:
            if not self.queue.empty():
                frame = self.queue.get()
                self.latest_frame = frame
                result = model(frame)
                labels = result.xyxy[0][:, -1].cpu().numpy()
                self.latest_detection = any(cls in [0, 2, 3, 5, 7] for cls in labels)

                if DISPLAY_VIDEO:
                    self.latest_frame = np.squeeze(result.render())

    def stop(self):
        self.running = False

# === MAIN THREAD ===
def main():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream.")
        return

    frame_queue = Queue(maxsize=10)

    grabber = FrameGrabber(cap, frame_queue)
    detector = Detector(frame_queue)

    grabber.start()
    detector.start()

    current_fps = LOW_FPS
    frame_interval = 1 / current_fps
    last_detection_time = None
    last_frame_time = time.time()
    chunk_start_time = datetime.now()
    video_writer = create_video_writer(current_fps)

    try:
        while True:
            now = time.time()

            # Use the latest frame from detector
            frame = detector.latest_frame
            detection = detector.latest_detection

            if frame is None:
                continue

            # Switch FPS if needed
            if detection:
                last_detection_time = now
                if current_fps != HIGH_FPS:
                    current_fps = HIGH_FPS
                    frame_interval = 1 / HIGH_FPS
                    video_writer.release()
                    video_writer = create_video_writer(HIGH_FPS)
                    print("[INFO] Detection found. Switching to HIGH FPS.")
            else:
                if last_detection_time and now - last_detection_time > NO_DETECTION_GRACE_PERIOD:
                    if current_fps != LOW_FPS:
                        current_fps = LOW_FPS
                        frame_interval = 1 / LOW_FPS
                        video_writer.release()
                        video_writer = create_video_writer(LOW_FPS)
                        print("[INFO] No detection. Switching to LOW FPS.")

            # Write video
            video_writer.write(frame)

            # Display video
            if DISPLAY_VIDEO:
                cv2.imshow("YOLOv5 Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Handle recording chunks
            if (datetime.now() - chunk_start_time).total_seconds() >= CHUNK_DURATION:
                video_writer.release()
                video_writer = create_video_writer(current_fps)
                chunk_start_time = datetime.now()

            time.sleep(max(0, frame_interval - (time.time() - now)))

    finally:
        grabber.stop()
        detector.stop()
        grabber.join()
        detector.join()
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print("Resources released, exiting.")

if __name__ == "__main__":
    main()
