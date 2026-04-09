from ultralytics import YOLO
import cv2
import os
from save_results import save_detections

def detect_webcam():
    model = YOLO("yolov8m.pt")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return
    
    os.makedirs("outputs/webcam_detection", exist_ok=True)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0
    
    output_path = "outputs/webcam_detection/webcam_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_id = 0
    
    print("Webcam started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fail to read frame from webcam.")
            break
        
        frame_id += 1
        
        results = model(frame, verbose=False)
        
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Webcam Detection", annotated_frame)
        out.write(annotated_frame)
        
        save_detections(
            results,
            source_name="webcam",
            frame_id=frame_id,
            output_csv="results/webcam_detections.csv"
        )
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
            
    print("Webcam detection completed.")
    print("Video saved to outputs/webcam_detection/webcam_output.mp4")
    print("CSV saved to results/webcam_detections.csv")
        
if __name__ == "__main__":
    detect_webcam()