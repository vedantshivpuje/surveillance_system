import cv2
from ultralytics import YOLO

class SpatialDetector:
    def __init__(self, model_path="models/weapon_yolov8.pt", conf_threshold=0.25):
        """
        Initializes YOLOv8 for weapon and object detection.
        Lowered conf_threshold to 0.25 for better real-time sensitivity.
        """
        try:
            self.model = YOLO(model_path)
            print(f"[INFO] Loaded custom model: {model_path}")
        except Exception as e:
            print(f"[WARN] Failed loading {model_path}. Falling back to default yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            
        self.conf = conf_threshold
        # Expanded label list to cover common weapon class variations
        self.danger_labels = [
            "gun", "guns", "knife", "knives", "weapon", "weapons", 
            "pistol", "handgun", "rifle", "firearm", "blade", "scissors"
        ]

    def process_frame(self, frame):
        """
        Input: OpenCV BGR Frame
        Returns: Annotated Frame, List of detected labels, Boolean flag for weapon alert
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        annotated_frame = results.plot()
        
        detected_labels = []
        weapon_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label_name = self.model.names[cls_id].lower()
            detected_labels.append(label_name)
            
            if label_name in self.danger_labels:
                weapon_detected = True

        return annotated_frame, detected_labels, weapon_detected