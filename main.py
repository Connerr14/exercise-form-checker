import cv2
from cam import VideoStream
from pose_tracker import PoseTracker
from connectToEsp import EspCommunication
import numpy as np

# Initialize modules
stream = VideoStream(0)
tracker = PoseTracker()

# MediaPipe landmark indices for the right and left leg
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27

RIGHT_SHOULDER = 12
LEFT_SHOULDER = 11

# Initialize and start the ESP listener
esp = EspCommunication(port=4210)
esp.start()


def EvaluateForceReadings (lForce_raw, rForce_raw):
    print("here")


# Set the "Smoothness" (Higher = smoother but slower response)
BUFFER_SIZE = 15

while True:
    # Get video
    success, img = stream.read_frame()
    if not success: continue

    # Grab the screen dimensions to fix the aspect ratio for our lean math!
    img_height, img_width, _ = img.shape

    # Finds the joints
    img = tracker.process_frame(img)

    # Extract Right Leg
    r_hip = tracker.get_landmark_coords(RIGHT_HIP)
    r_knee = tracker.get_landmark_coords(RIGHT_KNEE)
    r_ankle = tracker.get_landmark_coords(RIGHT_ANKLE)
    right_angle = tracker.calculate_angle(r_hip, r_knee, r_ankle)

    # Extract Left Leg
    l_hip = tracker.get_landmark_coords(LEFT_HIP)
    l_knee = tracker.get_landmark_coords(LEFT_KNEE)
    l_ankle = tracker.get_landmark_coords(LEFT_ANKLE)
    left_angle = tracker.calculate_angle(l_hip, l_knee, l_ankle)
    left_angle = tracker.calculate_angle(l_hip, l_knee, l_ankle)

    # Get shoulder metrics
    r_shoulder = tracker.get_landmark_coords(RIGHT_SHOULDER)
    l_shoulder = tracker.get_landmark_coords(LEFT_SHOULDER)

   # Ensure all required joints are visible before doing math
    if r_hip is not None and l_hip is not None and r_shoulder is not None and l_shoulder is not None:
        
        # Calculate Knee Angles
        right_angle = tracker.calculate_angle(r_hip, r_knee, r_ankle)
        left_angle = tracker.calculate_angle(l_hip, l_knee, l_ankle)
        
        asymmetry = abs(right_angle - left_angle)

        # Calculate Lean
        r_lean = tracker.calculate_torso_lean(r_shoulder, r_hip, img_width, img_height)
        l_lean = tracker.calculate_torso_lean(l_shoulder, l_hip, img_width, img_height)
        
        # Average the lean metrics
        torso_lean = (r_lean + l_lean) / 2

        # Hardware Data Integration
        sensor_data = esp.read_packet()
        if sensor_data:
            parts = sensor_data.split()
            # Check to ensure the packet isn't messed up before converting
            if len(parts) == 2:
                lForce_raw = int(parts[0])
                rForce_raw = int(parts[1])
                print(f"Left: {lForce_raw}, Right: {rForce_raw}")
                EvaluateForceReadings(lForce_raw, rForce_raw)
            else:
                print("Error getting packet: " + sensor_data)

        
        # Start with a clean slate for this frame
        active_warnings = []
        skeleton_color = (0, 255, 0) # Default Green (Good)

        # Evaluate Form
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

        if len(active_warnings) == 0:
            active_warnings.append("Good Form & Depth!")

        # Draw the skeleton with the final determined color
        img = tracker.draw_skeleton(img, line_color=skeleton_color)

        # Display the live stats
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
            # If no angle is detected, just draw a default white skeleton
            img = tracker.draw_skeleton(img)

    # Show cam result
    cv2.imshow("Squat Form Tracker", img)

    # Quit on q key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything safely
esp.stop()
stream.stop()
cv2.destroyAllWindows()