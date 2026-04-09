from ultralytics import YOLO
from save_results import save_detections
import os
import cv2

def detect_video(video_path):
    print("Current working directory:", os.getcwd())
    print("Checking video path:", video_path)
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    
    frame_id = 0
        
    model(
        video_path,
        save=True,
        project="outputs",
        name="video_detection",
        exist_ok=True
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        if frame_id % 5 != 0:
            continue
        
        resutls = model(frame, verbose=False)
        
        save_detections(
            resutls,
            source_name=os.path.basename(video_path),
            frame_id=frame_id,
            output_csv="results/video_detections.csv"
        )
    
    cap.release()
    
    print("Video detection and CSV saving completed.")
    print("Output saved in: output/video_detection")
    
if __name__ == "__main__":
    video_path = "data/input_videos/traffic.mp4"
    detect_video(video_path)