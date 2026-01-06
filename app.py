import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from supabase import create_client

# --- CREDENTIALS (Pulled from Streamlit Secrets) ---
URL = st.secrets["SUPABASE_URL"]
KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(URL, KEY)

# Load model (Using 'best.pt' as seen in your GitHub screenshot)
model = YOLO("best.pt") 

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Premolt Monitor", page_icon="🦀", layout="wide")

# Custom CSS for a clean, modern look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #1e2130;
        border: 1px solid #2ecc71;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricValue"] { color: #2ecc71 !important; font-size: 50px !important; }
    div[data-testid="stMetricLabel"] { color: #ffffff !important; font-size: 20px !important; }
    .stFileUploader { border: 2px dashed #2ecc71; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🦀 Premolt Real-Time Monitor")
st.caption("Advanced AI Analysis for Crab Molting Cycles")
st.divider()

# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Control Panel")
    uploaded_file = st.file_uploader("Upload Video Feed", type=["mp4", "mov", "avi"])
    metric_display = st.metric(label="ACTIVE PREMOLT COUNT", value=0)
    
    if not uploaded_file:
        st.info("Waiting for video input...")
        st.image("https://img.icons8.com/clouds/200/000000/crab.png")

with col1:
    video_placeholder = st.empty()

if uploaded_file:
    # Processing video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detection (Change classes=[1] to [0] if premolt is your first class)
        results = model.predict(frame, conf=0.4, classes=[1], verbose=False)
        count = len(results[0].boxes)

        # Custom Drawing (Clean Green Boxes)
        annotated_frame = frame.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (46, 204, 113), 3)

        # UI Updates
        metric_display.metric(label="ACTIVE PREMOLT COUNT", value=count)
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        # Database Sync
        try:
            supabase.table("counts").upsert({"label": "premolt_now", "value": count}).execute()
        except:
            pass

    cap.release()
