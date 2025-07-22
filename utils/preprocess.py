import os
import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Config
IMG_SIZE = (64, 64)
PPG_LENGTH = 100
IMG_DIR = "./data/images"
PPG_DIR = "./data/ppg_signals"
CSV_PATH = "./data/labels.csv"

# Preprocess PPG
def preprocess_ppg(ppg_path, target_length=PPG_LENGTH):
    df = pd.read_csv(ppg_path)  # header is auto-detected

    if 'ppg' not in df.columns:
        raise ValueError(f"'ppg' column not found in {ppg_path}")

    ppg = df['ppg'].astype(float).values.flatten()

    # Pad or truncate
    if len(ppg) < target_length:
        ppg = np.pad(ppg, (0, target_length - len(ppg)), mode='constant')
    else:
        ppg = ppg[:target_length]

    # Smooth + Normalize
    ppg = savgol_filter(ppg, window_length=9, polyorder=3)
    ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-6)

    return ppg


def preprocess_image(img_path, target_size=IMG_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

# Load and preprocess entire dataset
def load_dataset(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    images, ppgs, labels = [], [], []

    for idx, row in df.iterrows():
        # Safe and normalized paths
        img_path = os.path.normpath(os.path.join(IMG_DIR, row['image']))
        ppg_path = os.path.normpath(os.path.join(PPG_DIR, row['ppg']))

        # Debug info if files don't exist
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(ppg_path):
            raise FileNotFoundError(f"PPG file not found: {ppg_path}")

        # Load & preprocess
        img = preprocess_image(img_path)
        ppg = preprocess_ppg(ppg_path)

        images.append(img)
        ppgs.append(ppg)
        labels.append(row['label'])

    # Final arrays
    X_img = np.array(images)
    X_ppg = np.array(ppgs).reshape(-1, PPG_LENGTH, 1)
    
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(labels)
    y_cat = to_categorical(y)

    return X_img, X_ppg, y_cat, label_enc.classes_


