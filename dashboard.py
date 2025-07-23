# dashboard.py

import streamlit as st
import torch
import numpy as np
import time
import os
import sys

# Project root setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from model.snn_model import SNNDriverStateClassifier
from utils.preprocess import load_dataset

# Load class labels
class_labels = {
    0: "🟢 Awake",
    1: "⚠️ Drowsy",
    2: "❌ Distracted"
}

@st.cache_resource
def load_model():
    model = SNNDriverStateClassifier()
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "snn_model.pth"), map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_sample_data():
    X_img, X_ppg, y_cat, _ = load_dataset()
    return X_img, X_ppg, y_cat

# App layout
st.set_page_config(page_title="SafeSpikr Dashboard", layout="wide")
st.title("🚗 SafeSpikr Driver State Inference Dashboard")

# Load model and data
model = load_model()
X_img, X_ppg, y_cat = load_sample_data()

# Select sample index
index = st.slider("Select Sample Index", 0, len(X_img) - 1, 0)

# Display Image Message
st.subheader("🖼️ Driver Image")
st.info("Driver image has been extracted and processed successfully.")

# Display PPG (line chart)
st.subheader("📈 Simulated PPG Signal")
ppg_flat = X_ppg[index].reshape(-1)
st.line_chart(ppg_flat)

# Run inference
st.subheader("🧠 Driver State Prediction")

with torch.no_grad():
    img_tensor = torch.from_numpy(X_img[index]).permute(2, 0, 1).unsqueeze(0).float()
    ppg_tensor = torch.from_numpy(X_ppg[index]).unsqueeze(0).float()

    output = model(img_tensor, ppg_tensor)
    prediction = torch.argmax(output, dim=1).item()
    prob = torch.nn.functional.softmax(output, dim=1)[0, prediction].item()

state = class_labels.get(prediction, "Unknown")
st.success(f"### Predicted State: {state} ({prob*100:.2f}%)")
