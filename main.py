import cv2
import numpy as np
from cam import VideoStream
from pose_tracker import PoseTracker
from connectToEsp import EspCommunication
from data_handler import SquatDataCollector
import joblib
from squat_coach import SquatCoach
import pandas as pd
from collections import deque


ai_model = joblib.load('final_squat_model.joblib')

coach = SquatCoach()
LAST_AVG_ANGLE = 180
prediction_history = deque(maxlen=6)

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
    global LAST_WEIGHT_STATE, LAST_AVG_ANGLE
    angle_speed = 0.0
    img_height, img_width, _ = img.shape
    img = tracker.process_frame(img)

    # Landmark Extraction
    r_hip, r_knee, r_ankle = tracker.get_landmark_coords(RIGHT_HIP), tracker.get_landmark_coords(RIGHT_KNEE), tracker.get_landmark_coords(RIGHT_ANKLE)
    l_hip, l_knee, l_ankle = tracker.get_landmark_coords(LEFT_HIP), tracker.get_landmark_coords(LEFT_KNEE), tracker.get_landmark_coords(LEFT_ANKLE)
    r_shoulder, l_shoulder = tracker.get_landmark_coords(RIGHT_SHOULDER), tracker.get_landmark_coords(LEFT_SHOULDER)

    if all(v is not None for v in [r_hip, l_hip, r_shoulder, l_shoulder, r_knee, r_ankle]):
        # --- MATH & PHYSICS ---
        right_angle = tracker.calculate_angle(r_hip, r_knee, r_ankle)
        left_angle = tracker.calculate_angle(l_hip, l_knee, l_ankle)
        asymmetry = abs(right_angle - left_angle)
        torso_lean = (tracker.calculate_torso_lean(r_shoulder, r_hip, img_width, img_height) + 
                      tracker.calculate_torso_lean(l_shoulder, l_hip, img_width, img_height)) / 2

        current_avg_angle = (right_angle + left_angle) / 2
        angle_speed = current_avg_angle - LAST_AVG_ANGLE
        LAST_AVG_ANGLE = current_avg_angle

        # ESP Data
        sensor_data = esp.read_packet()
        squatForceReading = EvaluateForceReadings(sensor_data)
        if (squatForceReading != -1 and squatForceReading is not None):
            LAST_WEIGHT_STATE = checkWeightDistribution(squatForceReading[0], squatForceReading[1])

        # -- DUAL MODE LOGIC -
        
        if collector_obj.is_recording:
            # - MODE A: RECORDING (Red UI, Manual Warnings) -
            active_warnings = []
            if torso_lean > 10: active_warnings.append("Chest UP!")
            if asymmetry > 8: active_warnings.append("Uneven Squat!")
            if current_avg_angle >= 95: active_warnings.append("Go Lower")
            if 0 in LAST_WEIGHT_STATE: active_warnings.append("Check Weight!")

            # Save Snapshot (9-column CSV format)
            snapshot = [
                round(torso_lean, 2), round(asymmetry, 2), 
                round(right_angle, 2), round(left_angle, 2), 
                LAST_WEIGHT_STATE[0], LAST_WEIGHT_STATE[1], LAST_WEIGHT_STATE[2],
                "TRAINING_DATA",
                round(angle_speed, 2)
            ]
            collector_obj.collect(snapshot)

            # Rendering Recording UI
            img = tracker.draw_skeleton(img, line_color=(0, 0, 255))
            cv2.putText(img, "RECORDING DATA", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Lean: {int(torso_lean)} | Depth: {int(current_avg_angle)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            y_off = 150
            for w in active_warnings:
                cv2.putText(img, w, (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
                y_off += 50

        else:
            # --- MODE B: AI COACHING (Live AI Prediction) ---
            # Create Feature DataFrame
            features = pd.DataFrame([[
                round(torso_lean, 2), round(asymmetry, 2), round(right_angle, 2), round(left_angle, 2), 
                LAST_WEIGHT_STATE[0], LAST_WEIGHT_STATE[1], LAST_WEIGHT_STATE[2], angle_speed
            ]], columns=['lean', 'asymmetry', 'right_angle', 'left_angle', 'force_right', 'force_left', 'force_diff', 'angle_diff'])
            
            raw_pred = ai_model.predict(features)[0]
            prediction_history.append(raw_pred)
            stable_pred = max(set(prediction_history), key=prediction_history.count)

            coach.process_frame(stable_pred, raw_pred)

            # Rendering Coach UI 
            status_color = (0, 255, 0) if stable_pred == "At_Bottom" else (255, 255, 255)
            if "FIX" in coach.live_feedback: status_color = (0, 0, 255)
            
            img = tracker.draw_skeleton(img, line_color=status_color)
            cv2.putText(img, coach.live_feedback.upper(), (40, 100), cv2.FONT_HERSHEY_DUPLEX, 1.8, status_color, 4)
            cv2.putText(img, f"REPS: {coach.rep_count}", (40, img_height - 60), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 0), 5)
            if coach.last_verdict:
                cv2.putText(img, f"LAST: {coach.last_verdict}", (img_width - 600, img_height - 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 1)

          # - DYNAMIC PROGRESS GAUGE (Right Side) -
        avg_angle = int(current_avg_angle)
        
        # Goal Color Logic (Traffic Light)
        if avg_angle > 135:
            bar_color = (0, 0, 255)    # RED: Just started
        elif avg_angle > 95:
            bar_color = (0, 165, 255)  # ORANGE: Almost there
        else:
            bar_color = (0, 255, 0)    # GREEN: Goal Hit!

        # Gauge Dimensions
        bar_x = img_width - 100
        bar_top = 150
        bar_bottom = 450
        bar_height = bar_bottom - bar_top

        # 180° -> bar_bottom (empty)
        # 95°  -> bar_top (full)
        fill_y = int(np.interp(avg_angle, [95, 180], [bar_top, bar_bottom]))

        # Draw the Background (Dark Grey)
        cv2.rectangle(img, (bar_x, bar_top), (bar_x + 50, bar_bottom), (40, 40, 40), -1)
        
        # Draw the Progress Fill (From the calculated fill_y down to the bottom)
        cv2.rectangle(img, (bar_x, fill_y), (bar_x + 50, bar_bottom), bar_color, -1)

        # Draw the Target "Finish Line" at the top
        cv2.line(img, (bar_x - 10, bar_top), (bar_x + 60, bar_top), (255, 255, 255), 3)

        # LABELS
        cv2.putText(img, "GOAL", (bar_x - 15, bar_top - 20), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(img, f"{avg_angle} deg", (bar_x - 30, bar_bottom + 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

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