# app.py
import os
import cv2
import time
import numpy as np
import streamlit as st

from modules.spatial_detector import SpatialDetector
from modules.temporal_anomaly import TemporalAnomalyDetector
from modules.clip_recorder import ClipRecorder

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="🛡",
    layout="wide"
)

st.title("🛡 Automated Smart Video Surveillance & Anomaly Detector")
st.markdown("**Team:** Vishal (Spatial) | Vismit (Temporal) | Vedant (Pipeline & UI)")

# --- 2. Sidebar Controls ---
st.sidebar.header("🕹 System Controls")
video_source = st.sidebar.selectbox("Select Feed", ["Sample Video", "Webcam Feed"])
conf_thresh = st.sidebar.slider("YOLO Confidence Threshold", 0.3, 0.9, 0.5, 0.05)
motion_thresh = st.sidebar.slider("Anomaly Sensitivity", 10, 50, 25, 1)

# Start / Stop Toggle
run_system = st.sidebar.toggle("Start Monitoring", value=False)

# --- 3. Module Initialization (Cached) ---
@st.cache_resource
def init_system():
    # Adjust paths if your weight locations differ
    spatial = SpatialDetector(model_path="models/weapon_yolov8.pt")
    temporal = TemporalAnomalyDetector()
    recorder = ClipRecorder(output_dir="saved_alerts")
    return spatial, temporal, recorder

try:
    spatial_net, temporal_net, recorder = init_system()
except Exception as e:
    st.sidebar.error(f"Module load warning: {e}")
    # Fallback to direct init if class constructors don't take arguments
    spatial_net = SpatialDetector()
    temporal_net = TemporalAnomalyDetector()
    recorder = ClipRecorder()

# --- 4. Main Interface Tabs ---
tab_live, tab_vault = st.tabs(["🎥 Live Surveillance Stream", "📁 Incident Clip Vault"])

with tab_live:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Live Feed Feed")
        FRAME_WINDOW = st.empty()

    with col2:
        st.subheader("Real-Time Analytics")
        alert_placeholder = st.empty()
        anomaly_metric = st.metric(label="Anomaly Motion Score", value="0.0")
        detected_objects_box = st.empty()
        clip_save_status = st.empty()

    # --- 5. Main Processing Loop ---
    if run_system:
        # Determine video input
        video_path = 0 if video_source == "Webcam Feed" else "sample_data/fight_weapon.mp4"
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error(f"Cannot open video feed: {video_path}")
        else:
            frame_buffer = []

            while cap.isOpened() and run_system:
                ret, frame = cap.read()
                if not ret:
                    st.info("Video feed ended or source unreachable.")
                    break

                # Normalize resolution
                frame = cv2.resize(frame, (640, 480))
                frame_buffer.append(frame.copy())
                
                # Maintain rolling 3-second buffer (~20 fps)
                if len(frame_buffer) > 60:
                    frame_buffer.pop(0)

                # 1. Spatial Detection (Vishal)
                annotated_frame, labels, weapon_found = spatial_net.detect_frame(
                    frame, conf_threshold=conf_thresh
                )

                # 2. Temporal Anomaly Detection (Vismit)
                anomaly_score, is_anomaly = temporal_net.compute_motion_anomaly(
                    frame, motion_threshold=motion_thresh
                )

                # Update UI Analytics
                anomaly_metric.metric(label="Anomaly Motion Score", value=f"{anomaly_score:.2f}")
                detected_entities = ", ".join(set(labels)) if labels else "None"
                detected_objects_box.markdown(f"**Detected Entities:** `{detected_entities}`")

                # Threat evaluation & Alert handling
                if weapon_found or is_anomaly:
                    threat_types = []
                    if weapon_found:
                        threat_types.append("Weapon Detected!")
                    if is_anomaly:
                        threat_types.append("High Anomaly Motion!")

                    alert_msg = f"⚠ **ALERT:** {' '.join(threat_types)}"
                    alert_placeholder.error(alert_msg)

                    # Auto-save evidence clip using recent frame buffer
                    saved_path = recorder.save_alert_clip(frame_buffer)
                    if saved_path:
                        clip_save_status.success(f"💾 Clip saved: `{os.path.basename(saved_path)}`")
                else:
                    alert_placeholder.success("🟢 SYSTEM NORMAL - Monitoring Feed...")
                    clip_save_status.empty()

                # Render processed frame to UI
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
    else:
        alert_placeholder.info("System Standby. Turn on 'Start Monitoring' from the sidebar to begin.")

with tab_vault:
    st.subheader("Evidence Recordings")
    os.makedirs("saved_alerts", exist_ok=True)
    recorded_files = sorted(
        [f for f in os.listdir("saved_alerts") if f.endswith((".mp4", ".avi"))],
        reverse=True
    )

    if recorded_files:
        chosen_clip = st.selectbox("Select saved evidence clip to view:", recorded_files)
        if chosen_clip:
            st.video(os.path.join("saved_alerts", chosen_clip))
    else:
        st.info("No alert clips generated yet. Triggered events will display here.")
