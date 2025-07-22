import numpy as np
import pandas as pd
import cv2
import os

# Create PPG signal (same as before)
def generate_ppg_signal(sample_id, label, save_path):
    t = np.linspace(0, 10, 1000)
    noise = np.random.normal(0, 0.02, size=t.shape)
    
    if label == "alert":
        signal = 0.6 * np.sin(1.5 * t) + noise
    elif label == "drowsy":
        signal = 0.3 * np.sin(0.5 * t) + noise
    else:  # stressed
        signal = 0.8 * np.sin(2.5 * t) + noise

    df = pd.DataFrame({"time": t, "ppg": signal})
    df.to_csv(f"{save_path}/ppg_{sample_id}.csv", index=False)

# Create clearer synthetic image
def generate_image(sample_id, label, save_path):
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255  # White background

    if label == "alert":
        # Draw bright green circle
        cv2.circle(img, (32, 32), 15, (0, 255, 0), -1)

    elif label == "drowsy":
        # Draw drooping eye using arc
        cv2.ellipse(img, (32, 32), (20, 10), 0, 0, 180, (128, 128, 0), 2)

    else:  # stressed
        # Draw zig-zag red pattern
        pts = np.array([[10, 50], [20, 30], [30, 50], [40, 30], [50, 50]], np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 255), thickness=2)

    cv2.imwrite(f"{save_path}/img_{sample_id}.png", img)

# Main dataset generator
def generate_dataset(num_samples=200):
    labels = ["alert", "drowsy", "stressed"]
    image_path = "data/images"
    ppg_path = "data/ppg_signals"
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(ppg_path, exist_ok=True)
    
    records = []

    for i in range(num_samples):
        label = np.random.choice(labels)
        generate_image(i, label, image_path)
        generate_ppg_signal(i, label, ppg_path)
        records.append({
            "id": i,
            "image": f"img_{i}.png",
            "ppg": f"ppg_{i}.csv",
            "label": label
        })
    
    df = pd.DataFrame(records)
    df.to_csv("data/labels.csv", index=False)
    print(f"✅ Synthetic dataset of {num_samples} samples generated!")

if __name__ == "__main__":
    generate_dataset(num_samples=200)
