import cv2
import numpy as np
from cam import VideoStream
from pose_tracker import PoseTracker
from connectToEsp import EspCommunication
from data_handler import SquatDataCollector

# --- Initialization ---
stream = VideoStream(0)
tracker = PoseTracker()
esp = EspCommunication(port=4210)
esp.start()

# Initialize the collector (10s delay logic lives inside this object)
collector = SquatDataCollector(filename="squat_dataset.csv", delay=10.0)

# Landmark Constants
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_SHOULDER, LEFT_SHOULDER = 12, 11
BUFFER_SIZE = 3
BUFFER_RIGHT = []
BUFFER_LEFT = []
BUFFER_COUNT = 0
LAST_WEIGHT_STATE = [1, 1, 1] # Default to "Good"

def EspReadingsBuffer (lForce_raw, rForce_raw):
    global BUFFER_COUNT
    BUFFER_LEFT.append(lForce_raw)
    BUFFER_RIGHT.append(rForce_raw)
    BUFFER_COUNT += 1

    if (BUFFER_COUNT >= BUFFER_SIZE):
        averageDifferenceR = 0
        averageDifferenceL = 0
        # Get the average difference in reading between the packets
        for r in range(len(BUFFER_RIGHT) - 1):
            RDifference = BUFFER_RIGHT[r+1] - BUFFER_RIGHT[r]
            averageDifferenceR += RDifference
        
        for l in range(len(BUFFER_LEFT) - 1):
            LDifference = BUFFER_LEFT[l+1] - BUFFER_LEFT[l]
            averageDifferenceL += LDifference

        averageDifferenceR = averageDifferenceR/(BUFFER_SIZE - 1)
        averageDifferenceL = averageDifferenceL/(BUFFER_SIZE - 1)

        avgReadingTuple = (averageDifferenceR, averageDifferenceL)
        BUFFER_COUNT = 0
        BUFFER_LEFT.clear()
        BUFFER_RIGHT.clear()

        return avgReadingTuple
    
    else:
        # The buffer is not full yet
        return -1


def EvaluateForceReadings(sensor_data):
    if sensor_data:
        parts = sensor_data.split()
        if len(parts) == 2:
            lForce_raw, rForce_raw = int(parts[0]), int(parts[1])
            readingTuple = EspReadingsBuffer(lForce_raw, rForce_raw)
            if (readingTuple == -1):
                return -1
            else:
                return readingTuple
        return -1
    
def checkWeightDistribution (right_difference_avg, left_difference_avg):  
    weightDistributionR = 1
    weightDistributionL = 1
    weightDistributionDifference = 1
    cutOffValue = 1000
    
    if right_difference_avg > cutOffValue:
        weightDistributionR = 0
    if left_difference_avg > cutOffValue:
        weightDistributionL = 0 
    if abs(left_difference_avg - right_difference_avg) > cutOffValue:
        weightDistributionDifference = 0

    return [weightDistributionR, weightDistributionL, weightDistributionDifference]



def analyze_squat_form(img, collector_obj):
    """Handles the math, hardware data, and UI drawing for a single frame."""
    global LAST_WEIGHT_STATE
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
    if all(v is not None for v in [r_hip, l_hip, r_shoulder, l_shoulder, r_knee, r_ankle, l_knee, l_ankle]):
        # Math
        right_angle = tracker.calculate_angle(r_hip, r_knee, r_ankle)
        left_angle = tracker.calculate_angle(l_hip, l_knee, l_ankle)
        asymmetry = abs(right_angle - left_angle)
        
        r_lean = tracker.calculate_torso_lean(r_shoulder, r_hip, img_width, img_height)
        l_lean = tracker.calculate_torso_lean(l_shoulder, l_hip, img_width, img_height)
        torso_lean = (r_lean + l_lean) / 2

        # Hardware Integration
        sensor_data = esp.read_packet()
        squatForceReading = EvaluateForceReadings(sensor_data)
       # Only update the state when the buffer outputs new averages
        if (squatForceReading != -1 and squatForceReading is not None):
            LAST_WEIGHT_STATE = checkWeightDistribution(squatForceReading[0], squatForceReading[1])

        # Form Evaluation
        active_warnings = []
        skeleton_color = (0, 255, 0) # Green

        if torso_lean > 10:
            active_warnings.append("Chest UP! Don't lean.")
            skeleton_color = (0, 195, 255) # Orange
        if asymmetry > 8: 
            active_warnings.append("Uneven Squat!")
            skeleton_color = (0, 255, 255) # Yellow
        if right_angle >= 90 or left_angle >= 90:
            active_warnings.append("Go Lower")
            if len(active_warnings) == 1: 
                skeleton_color = (0, 0, 255) # Red
        if 0 in LAST_WEIGHT_STATE:
            active_warnings.append("Use even weight distribution.")

        if not active_warnings:
            active_warnings.append("Good Form & Depth!")

        # If the collector is in recording mode, send it the data
        if collector_obj.is_recording:
            snapshot = [
                round(torso_lean, 2), 
                round(asymmetry, 2), 
                round(right_angle, 2), 
                round(left_angle, 2), 
                LAST_WEIGHT_STATE[0], LAST_WEIGHT_STATE[1], LAST_WEIGHT_STATE[2],
                "TRAINING_DATA"
            ]
            # This function handles the 10s start delay and the 10s end discard
            collector_obj.collect(snapshot)


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
# Initialize the collector
collector = SquatDataCollector(delay=10.0)

while True:
    success, img = stream.read_frame()
    if not success: continue

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
            if not collector.is_recording:
                collector.start()
            else:
                collector.stop()

    # Process the frame logic
    img = analyze_squat_form(img, collector)

    cv2.imshow("Squat Form Tracker", img)

    if key == ord('q'):
        break

# Cleanup
esp.stop()
stream.stop()
cv2.destroyAllWindows()