import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from supabase import create_client
import time

# --- SETUP ---
# Replace with your Supabase credentials
URL = "https://jqcuidwpxrrlhweuyguu.supabase.co"
KEY = "sb_publishable_SfTyHylwU2HTTRGFOPVByQ_RAobUn4r"
supabase = create_client(URL, KEY)

# Load your custom model (ensure 'crab_model.pt' is in the same folder)
model = YOLO("best.pt") 

st.set_page_config(page_title="Crab Detection Dashboard", layout="wide")
st.title("🦀 Crab & Premolt Detection")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
# Specify your class IDs (e.g., 0: Crab, 1: Premolt)
target_classes = [0, 1] 

# --- VIDEO UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload a Video of Crabs", type=["mp4", "mov", "avi"])

# --- MAIN INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()  # Where the frames will appear

with col2:
    st.subheader("Current Statistics")
    crab_metric = st.empty()
    premolt_metric = st.empty()

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run Custom YOLO Inference
        results = model.predict(frame, conf=conf_threshold, classes=target_classes)
        
        # 2. Extract Counts
        # Assuming Class 0 = Crab, Class 1 = Premolt
        counts = results[0].boxes.cls.tolist()
        num_crabs = counts.count(0)
        num_premolt = counts.count(1)

        # 3. Update Supabase (Public Access)
        try:
            supabase.table("counts").upsert({"label": "crab_count", "value": num_crabs}).execute()
            supabase.table("counts").upsert({"label": "premolt_count", "value": num_premolt}).execute()
        except Exception as e:
            pass # Ignore DB errors to keep video smooth

        # 4. Draw Bounding Boxes
        annotated_frame = results[0].plot()
        
        # 5. Display on Web
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        crab_metric.metric("Total Crabs", num_crabs)
        premolt_metric.metric("Premolt Detected", num_premolt)

    cap.release()
    st.success("Processing Complete!")
else:
    st.info("Please upload a video file in the sidebar to begin detection.")
