import streamlit as st
import cv2
import tempfile
import pandas as pd
import pytz
from datetime import datetime
from ultralytics import YOLO
from supabase import create_client

# --- CREDENTIALS ---
URL = st.secrets["SUPABASE_URL"]
KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(URL, KEY)

model = YOLO("best.pt") 

# --- UI SETUP ---
# initial_sidebar_state="collapsed" hides it on desktop by default
st.set_page_config(page_title="Premolt Monitor", layout="wide", initial_sidebar_state="collapsed")

# CSS to completely hide the sidebar toggle on mobile for a cleaner look
st.markdown("<style>#MainMenu {visibility: hidden;} [data-testid='collapsedControl'] {display: none;}</style>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🦀 Premolt Monitor")

# --- TOP CONTROLS (Replacing Sidebar) ---
t_col1, t_col2, t_col3 = st.columns([1, 1, 1])
with t_col1:
    uploaded_file = st.file_uploader("1. Upload Video", type=["mp4", "mov", "avi"])
with t_col2:
    conf_threshold = st.slider("2. Confidence", 0.0, 1.0, 0.45)
with t_col3:
    # Set your local timezone (e.g., 'Asia/Bangkok', 'America/New_York')
    tz_choice = st.selectbox("3. Timezone", pytz.all_timezones, index=pytz.all_timezones.index('Asia/Bangkok'))

# --- DASHBOARD ---
st.divider()
m_col1, m_col2 = st.columns([2, 1])

with m_col2:
    st.subheader("📊 Live Data")
    metric_display = st.metric(label="Current Count", value=0)
    time_display = st.empty() # Placeholder for local time
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.download_button("📥 Download Log", df.to_csv(index=False), "data.csv", "text/csv")

with m_col1:
    video_placeholder = st.empty()

# --- PROCESSING ---
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=conf_threshold, classes=[1], verbose=False)
        count = len(results[0].boxes)
        
        # Get correct local time
        local_tz = pytz.timezone(tz_choice)
        local_time = datetime.now(local_tz).strftime("%H:%M:%S")

        # UI Updates
        metric_display.metric(label="PREMOLTS DETECTED", value=count)
        time_display.info(f"🕒 Local Time: {local_time}")
        
        # Render clean green boxes
        annotated_frame = frame.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        st.session_state.history.append({"time": local_time, "count": count})

        try:
            supabase.table("counts").upsert({"label": "premolt_now", "value": count}).execute()
        except: pass

    cap.release()
