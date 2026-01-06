import streamlit as st
from supabase import create_client
import time

st.set_page_config(page_title="Live Counter", page_icon="📊")

# Setup Supabase
URL = st.secrets["SUPABASE_URL"]
KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(URL, KEY)

st.title("🛰️ Real-Time Object Detection Dashboard")
st.write("This data is being updated live from a remote YOLO sensor.")

# Create a placeholder for the number
placeholder = st.empty()

while True:
    # Fetch data from Supabase
    response = supabase.table("counts").select("value").eq("label", "person_count").execute()
    
    if response.data:
        current_val = response.data[0]['value']
        
        with placeholder.container():
            st.metric(label="Current Person Count", value=current_val)
            st.info(f"Last updated: {time.strftime('%H:%M:%S')}")
    
    # Refresh every 2 seconds to avoid hitting API limits
    time.sleep(2)