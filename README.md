# Real-Time Object Detection System using YOLO

This project implements a real-time object detection pipeline using YOLO (You Only Look Once), supporting image, video, and webcam inputs.

---

## Features

- Object detection on static images  
- Object detection on video files  
- Real-time object detection using webcam  
- Automatic saving of detection results to CSV  
- Basic data analysis and visualization of detection outputs  

---

## Tech Stack

- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- pandas  
- matplotlib  

---

## Project Structure

```text
yolo-object-detection-project/
├── src/
├── data/
├── results/
├── outputs/
├── requirements.txt
└── README.md

---
## Installation

git clone https://github.com/pvughan/yolo-object-detection-project.git
cd yolo-object-detection-project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

--- 
## Usage

### Detect Image
python src/detect_image.py

### Detect Video
python src/detect_video.py

### Detect Webcam
python src/detect_webcam.py

### Analyze Results
python src/analyze_results.py

--- 
## Analysis
## 📊 Analysis

The analysis of detection results reveals clear patterns in object distribution and model behavior across different input types.

For image-based detection, the model successfully identifies dominant objects in traffic scenes, with cars being the most frequently detected class. This aligns well with the expected context of the input data.

In video scenarios, the model maintains consistent detection performance across frames, capturing variations in object density over time. The results indicate that object frequency fluctuates depending on scene dynamics, such as traffic flow.

The model demonstrates higher confidence when detecting larger and more visually distinct objects, such as buses and cars, while smaller objects like persons and motorcycles tend to have lower confidence scores.

In real-time webcam detection, the model shows increased instability compared to image and video inputs. It occasionally misclassifies objects, particularly when dealing with low lighting, motion blur, or visually similar shapes. This highlights the limitations of the model in real-world, dynamic environments.

Overall, the analysis shows that while the model performs well on structured inputs, its performance is sensitive to environmental conditions in real-time scenarios.

Additionally, the model tends to misclassify certain objects with similar shapes, indicating limitations in feature discrimination for fine-grained categories.

--- Future Improvements
- Save bounding box coordinates for deeper analysis
- Build a dashboard for visualization
- Fine-tun YOLO model on custom datasets
- Optimaize real-time inference performance

--
## Note
This project focuses on building a complete computer vision pipeline, including detection, data logging, and basic analysis.