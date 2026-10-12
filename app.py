# app.py
import os
import cv2
import time
import numpy as np
import streamlit as st

from modules.clip_recorder import ClipRecorder
from modules.spatial_detector import SpatialDetector
from modules.temporal_anomaly import TemporalAnomalyDetector

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="🛡",
    layout="wide"
)

st.title("🛡 Automated Smart Video Surveillance & Anomaly Detector")
st.caption("Team: Vishal (Spatial Detection) | Vismit (Temporal Anomaly) | Vedant (Pipeline & UI)")

# 2. Sidebar Controls
st.sidebar.header("🕹 System Controls")
video_source = st.sidebar.selectbox("Select Feed Source", ["Sample Video", "Webcam Feed"])
conf_thresh = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
motion_thresh = st.sidebar.slider("Anomaly Sensitivity", 5, 100, 25, 5)

run_system = st.sidebar.toggle("Start Surveillance Feed", value=False)

# 3. Model & Recorder Initialization (Cached for performance)
@st.cache_resource
def load_modules():
    spatial = SpatialDetector(model_path="models/weapon_yolov8.pt")
    temporal = TemporalAnomalyDetector()
    rec = ClipRecorder(output_dir="saved_alerts")
    return spatial, temporal, rec

spatial_net, temporal_net, recorder = load_modules()

# 4. Main Display Layout
tab_live, tab_history = st.tabs(["🎥 Live Surveillance Feed", "📁 Recorded Evidence Vault"])

with tab_live:
    col_feed, col_stats = st.columns([2.5, 1])

    with col_feed:
        st.subheader("Live Camera Stream")
        FRAME_WINDOW = st.empty()

    with col_stats:
        st.subheader("Real-Time Diagnostics")
        status_card = st.empty()
        metric_score = st.empty()
        entities_card = st.empty()
        recent_clip_card = st.empty()

    # Video Processing Loop
    if run_system:
        video_target = 0 if video_source == "Webcam Feed" else "sample_data/fight_weapon.mp4"
        cap = cv2.VideoCapture(video_target)

        if not cap.isOpened():
            st.error(f"Failed to access source: {video_target}")
        else:
            frame_buffer = []

            while cap.isOpened() and run_system:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video stream completed or interrupted.")
                    break

                frame = cv2.resize(frame, (640, 480))
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > 60:  # Maintain ~3 seconds buffer at 20 FPS
                    frame_buffer.pop(0)

                # Module 1: Spatial Detection (Vishal)
                annotated_frame, labels, weapon_found = spatial_net.detect_frame(frame, conf_threshold=conf_thresh)

                # Module 2: Temporal Anomaly (Vismit)
                anomaly_score, is_anomaly = temporal_net.compute_motion_anomaly(frame, motion_threshold=motion_thresh)

                # Metrics Update
                metric_score.metric("Motion Anomaly Score", f"{anomaly_score:.2f}")
                labels_str = ", ".join(set(labels)) if labels else "None"
                entities_card.markdown(f"**Detected Objects:** `{labels_str}`")

                # Threat Logic
                if weapon_found or is_anomaly:
                    threat_reasons = []
                    if weapon_found:
                        threat_reasons.append("Weapon Detected")
                    if is_anomaly:
                        threat_reasons.append("Irregular Motion Detected")

                    status_card.error(f"🚨 **ALERT TRIGGERED:** {' & '.join(threat_reasons)}")

                    # Trigger recording
                    saved_path = recorder.save_alert_clip(frame_buffer)
                    if saved_path:
                        recent_clip_card.success(f"Evidence clip generated: `{os.path.basename(saved_path)}`")
                else:
                    status_card.success("🟢 Monitoring Normal — Area Secure")

                # Frame Output
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
    else:
        status_card.info("System on Standby. Toggle 'Start Surveillance Feed' in the sidebar.")

with tab_history:
    st.subheader("Saved Security Recordings")
    if os.path.exists("saved_alerts"):
        clips = sorted(os.listdir("saved_alerts"), reverse=True)
        if clips:
            selected_clip = st.selectbox("Select Clip to Inspect", clips)
            if selected_clip:
                st.video(os.path.join("saved_alerts", selected_clip))
        else:
            st.info("No recorded incident clips yet.")
