import streamlit as st
import cv2
import tempfile
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from supabase import create_client

# --- CREDENTIALS ---
URL = st.secrets["SUPABASE_URL"]
KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(URL, KEY)

# Load model
model = YOLO("best.pt") 

# --- UI SETUP ---
st.set_page_config(page_title="Premolt Monitor", page_icon="🦀", layout="wide")

# Persistent state for results (to allow download)
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

# --- MAIN DASHBOARD ---
st.title("🦀 Premolt Monitor")

col1, col2 = st.columns([3, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    st.subheader("Live Stats")
    metric_display = st.metric(label="PREMOLTS DETECTED", value=0)
    
    # Download Section
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Detection Log",
            data=csv,
            file_name=f"crab_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# --- PROCESSING LOGIC ---
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. YOLO Detection with dynamic confidence
        results = model.predict(frame, conf=conf_threshold, classes=[1], verbose=False)
        count = len(results[0].boxes)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 2. Draw Custom Boxes & Real-Time Timestamp
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, f"LIVE: {timestamp}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_val = float(box.conf[0])
            # Green Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (46, 204, 113), 3)
            # Small Confidence Label (Optional, very clean)
            cv2.putText(annotated_frame, f"{conf_val:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 204, 113), 2)

        # 3. Log Data for Download
        st.session_state.history.append({"timestamp": timestamp, "count": count})

        # 4. UI Updates
        metric_display.metric(label="PREMOLTS DETECTED", value=count)
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        # 5. Database Sync
        try:
            supabase.table("counts").upsert({"label": "premolt_now", "value": count}).execute()
        except: pass

    cap.release()
