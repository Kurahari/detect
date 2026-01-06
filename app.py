import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from supabase import create_client

# --- CREDENTIALS ---
URL = st.secrets["SUPABASE_URL"]
KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(URL, KEY)

# Load your custom model
model = YOLO("crab_model.pt") 

# --- MOBILE UI SETUP ---
st.set_page_config(page_title="Premolt Tracker", layout="centered")

# Custom CSS to make the metrics look better on mobile
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 40px; color: #2ecc71; }
    .stApp { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦀 Premolt Monitor")

# File uploader (Simplified for mobile)
uploaded_file = st.file_uploader("Tap to Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # Large metric for mobile visibility
    metric_placeholder = st.empty()
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Run Detection (assuming class 1 is Premolt)
        # Change 'classes=[1]' to the index of your premolt class
        results = model.predict(frame, conf=0.5, classes=[1], verbose=False)
        
        premolt_count = len(results[0].boxes)

        # 2. Custom Drawing (Green Boxes, No Labels)
        annotated_frame = frame.copy()
        for box in results[0].boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw Green Box (BGR: 0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 3. Update Public Dashboard
        try:
            supabase.table("counts").upsert({"label": "premolt_now", "value": premolt_count}).execute()
        except:
            pass

        # 4. Display
        metric_placeholder.metric("Premolts Detected", premolt_count)
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
