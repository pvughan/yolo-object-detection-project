import pandas as pd
import os

def save_detections(results, source_name, output_csv="results/detections.csv", frame_id=None):
    data = []

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = round(float(box.conf[0]), 4)
            class_name = r.names[class_id]

            row = {
                "source": source_name,
                "class": class_name,
                "confidence": confidence
            }

            if frame_id is not None:
                row["frame"] = frame_id

            data.append(row)

    if not data:
        print("No detections to save.")
        return

    df = pd.DataFrame(data)

    os.makedirs("results", exist_ok=True)

    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)

    print(f"Saved detections to {output_csv}")