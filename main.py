import cv2
import numpy as np
from cam import VideoStream
from pose_tracker import PoseTracker
from connectToEsp import EspCommunication

# --- Initialization ---
stream = VideoStream(0)
tracker = PoseTracker()
esp = EspCommunication(port=4210)
esp.start()

# Landmark Constants
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_SHOULDER, LEFT_SHOULDER = 12, 11
BUFFER_SIZE = 15

def EvaluateForceReadings(sensor_data):
    if sensor_data:
        parts = sensor_data.split()
    if len(parts) == 2:
        lForce_raw, rForce_raw = int(parts[0]), int(parts[1])
    else:
        print(f"Error getting packet: {sensor_data}")
    # TO-DO: Analyze readings


def analyze_squat_form(img):
    """Handles the math, hardware data, and UI drawing for a single frame."""
    img_height, img_width, _ = img.shape
    img = tracker.process_frame(img)

    # 1. Landmark Extraction
    r_hip = tracker.get_landmark_coords(RIGHT_HIP)
    r_knee = tracker.get_landmark_coords(RIGHT_KNEE)
    r_ankle = tracker.get_landmark_coords(RIGHT_ANKLE)
    l_hip = tracker.get_landmark_coords(LEFT_HIP)
    l_knee = tracker.get_landmark_coords(LEFT_KNEE)
    l_ankle = tracker.get_landmark_coords(LEFT_ANKLE)
    r_shoulder = tracker.get_landmark_coords(RIGHT_SHOULDER)
    l_shoulder = tracker.get_landmark_coords(LEFT_SHOULDER)

    # Main Analysis Logic
    if all(v is not None for v in [r_hip, l_hip, r_shoulder, l_shoulder]):
        # Math
        right_angle = tracker.calculate_angle(r_hip, r_knee, r_ankle)
        left_angle = tracker.calculate_angle(l_hip, l_knee, l_ankle)
        asymmetry = abs(right_angle - left_angle)
        
        r_lean = tracker.calculate_torso_lean(r_shoulder, r_hip, img_width, img_height)
        l_lean = tracker.calculate_torso_lean(l_shoulder, l_hip, img_width, img_height)
        torso_lean = (r_lean + l_lean) / 2

        # Hardware Integration
        sensor_data = esp.read_packet()
        EvaluateForceReadings(sensor_data)

        # Form Evaluation
        active_warnings = []
        skeleton_color = (0, 255, 0) # Green

        if torso_lean > 10:
            active_warnings.append("Chest UP! Don't lean.")
            skeleton_color = (0, 195, 255) # Orange
        if asymmetry > 8: 
            active_warnings.append("Uneven Squat! Shift weight.")
            skeleton_color = (0, 255, 255) # Yellow
        if right_angle >= 100 or left_angle >= 100:
            active_warnings.append("Go Lower")
            if len(active_warnings) == 1: 
                skeleton_color = (0, 0, 255) # Red
        if not active_warnings:
            active_warnings.append("Good Form & Depth!")

        # UI Rendering
        img = tracker.draw_skeleton(img, line_color=skeleton_color)
        cv2.putText(img, f"Lean: {int(torso_lean)} deg", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, skeleton_color, 3, cv2.LINE_AA)
        cv2.putText(img, f"Depth: {int((right_angle + left_angle)/2)} deg", (30, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, skeleton_color, 3, cv2.LINE_AA)
        
        y_offset = 200
        for warning in active_warnings:
            cv2.putText(img, warning, (30, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, skeleton_color, 4, cv2.LINE_AA)
            y_offset += 70
    else:
        # Fallback if joints aren't found
        img = tracker.draw_skeleton(img)
        
    return img

# --------------- MAIN EXECUTION LOOP -----------------
while True:
    success, img = stream.read_frame()
    if not success: continue

    # Process the frame logic
    img = analyze_squat_form(img)

    cv2.imshow("Squat Form Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
esp.stop()
stream.stop()
cv2.destroyAllWindows()