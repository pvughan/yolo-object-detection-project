from ultralytics import YOLO
from save_results import save_detections
import os

def detect_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    model = YOLO("yolov8n.pt")
    results = model(
        image_path,
        save=True,
        project="outputs",
        name="image_detection",
        exist_ok=True
    )
    save_detections(
        results,
        source_name=os.path.basename(image_path),
        output_csv="results/image_detections.csv"
    )
    
    print("Image detection and CSV saving completed!")
    print("Output save in: output/image_detection")
    
if __name__ == "__main__":
    image_path = "data/input_images/bogotatraffic.jpg"
    detect_image(image_path)