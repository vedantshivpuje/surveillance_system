import cv2
from modules.spatial_detector import SpatialDetector

# 1. Load Vishal's Spatial Detector module using your trained weights
print("[INFO] Initializing Spatial Detector...")
detector = SpatialDetector(model_path="models/weapon_yolov8.pt", conf_threshold=0.5)

# 2. Access your computer's webcam (0 = primary camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Ensure no other app is using it.")
    exit()

print("[INFO] Webcam active. Press 'q' in the video window to stop testing.")

# 3. Stream loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame.")
        break

    # Pass frame to spatial detector module
    annotated_frame, detected_labels, weapon_found = detector.process_frame(frame)

    # Print detection output to VS Code Terminal
    if weapon_found:
        print(f"⚠️ WEAPON DETECTED! Labels: {detected_labels}")
    else:
        print(f"Status Normal | Detected: {detected_labels}")

    # Display real-time output in a pop-up window
    cv2.imshow("Vishal - Spatial Detector Local Test", annotated_frame)

    # Stop test if user presses 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print("[INFO] Test completed.")