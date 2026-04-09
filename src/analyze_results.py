import pandas as pd
import matplotlib.pyplot as plt


def analyze(csv_path, title="Analysis"):
    print(f"\n===== {title} =====")

    df = pd.read_csv(csv_path)

    print("\nTotal detections:", len(df))

    print("\nObject count by class:")
    counts = df["class"].value_counts()
    print(counts)

    print("\nAverage confidence by class:")
    conf = df.groupby("class")["confidence"].mean()
    print(conf)

    plt.figure()
    counts.plot(kind="bar")
    plt.title(f"{title} - Object Frequency")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze("results/image_detections.csv", title="Image Detection")

    analyze("results/video_detections.csv", title="Video Detection")

    try:
        analyze("results/webcam_detections.csv", title="Webcam Detection")
    except:
        print("\n(No webcam CSV found)")