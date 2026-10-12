# modules/clip_recorder.py
import cv2
import os
import time
from datetime import datetime

class ClipRecorder:
    def __init__(self, output_dir="saved_alerts", cooldown_seconds=5):
        self.output_dir = output_dir
        self.cooldown_seconds = cooldown_seconds
        self.last_saved_time = 0
        os.makedirs(output_dir, exist_ok=True)

    def save_alert_clip(self, frame_buffer, fps=20):
        current_time = time.time()
        # Prevent spamming clip generation on continuous alert frames
        if current_time - self.last_saved_time < self.cooldown_seconds:
            return None

        if not frame_buffer:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        height, width, _ = frame_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        for frame in frame_buffer:
            out.write(frame)

        out.release()
        self.last_saved_time = current_time
        return filepath
