import cv2
import numpy as np

class TemporalAnomalyDetector:
    def __init__(self, motion_threshold=25.0):
        self.prev_gray = None
        self.motion_threshold = motion_threshold

    def compute_motion_anomaly(self, frame):
        """
        Calculates Optical Flow motion intensity between consecutive frames.
        Returns: anomaly_score (float 0.0 - 100.0), is_anomaly (bool)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, False

        # Compute Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Calculate magnitude of 2D motion vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_motion = np.mean(magnitude) * 10.0

        self.prev_gray = gray

        # Cap score between 0 and 100
        anomaly_score = min(float(mean_motion), 100.0)
        is_anomaly = anomaly_score > self.motion_threshold

        return round(anomaly_score, 2), is_anomaly

    def reset(self):
        """Reset tracking state when video stream restarts"""
        self.prev_gray = None