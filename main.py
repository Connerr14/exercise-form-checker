# Importing the necessary libraries 
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

# Loading in the model file
ai_model = joblib.load('final_squat_model.joblib')

# Creating an instance of the SquatCoach class
coach = SquatCoach()

# Setting up a buffer for the prediction history
prediction_history = deque(maxlen=6)

# Initializing a video stream, tracker object, and a esp communication object
stream = VideoStream(0)
tracker = PoseTracker()
esp = EspCommunication(port=4210)

# Starting esp communication
esp.start()

# Initialize the collector (For data recording)
collector = SquatDataCollector(filename="squat_dataset.csv", delay=10.0)

# Initializing Landmark Constant Variables (Mappings of the human body, which are connected with lines of differing length depending on the users height (later))
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_SHOULDER, LEFT_SHOULDER = 12, 11
LAST_WEIGHT_STATE = [1, 1, 1]
LAST_AVG_ANGLE = 180.0

# Setting up buffer variables
BUFFER_SIZE = 3
BUFFER_RIGHT = []
BUFFER_LEFT = []
BUFFER_COUNT = 0

""" A function to save the espReadings in a buffer, and returns the buffers when they are full """
def EspReadingsBuffer (lForce_raw, rForce_raw):
    # Using a buffer count var to track how many packets are in reserve
    global BUFFER_COUNT

    # Appending the new readings to their respective buffer lists, and incrementing the buffer count
    BUFFER_LEFT.append(lForce_raw)
    BUFFER_RIGHT.append(rForce_raw)
    BUFFER_COUNT += 1
    
    # Check if the buffer is full and ready to be processed
    if (BUFFER_COUNT >= BUFFER_SIZE):

        # Clear the variables that will be updated
        averageDifferenceR = 0
        averageDifferenceL = 0

        # Get the average difference in reading between the packets (For the left foot frs readings)
        for r in range(len(BUFFER_RIGHT) - 1):
            RDifference = BUFFER_RIGHT[r+1] - BUFFER_RIGHT[r]
            averageDifferenceR += RDifference
        
        # Get the average difference in reading between the packets (For the right foot frs readings)
        for l in range(len(BUFFER_LEFT) - 1):
            LDifference = BUFFER_LEFT[l+1] - BUFFER_LEFT[l]
            averageDifferenceL += LDifference

        # Getting the average difference between the packet data, and storing them in a tuple
        averageDifferenceR = averageDifferenceR/(BUFFER_SIZE - 1)
        averageDifferenceL = averageDifferenceL/(BUFFER_SIZE - 1)

        avgReadingTuple = (averageDifferenceR, averageDifferenceL)

        # Resetting the buffer variables
        BUFFER_COUNT = 0
        BUFFER_LEFT.clear()
        BUFFER_RIGHT.clear()

        # Returning the tuple with the new averages
        return avgReadingTuple
    # If the buffer is not full yet, return -1
    else:
        return -1

""" A function to process the FSR readings, returning them in a tuple format """
def EvaluateForceReadings(sensor_data):
    # If the sensor data packet was received, split it, and get the left and right fsr readings
    if sensor_data:
        # Splitting the data on the space
        parts = sensor_data.split()
        if len(parts) == 2:
            # Save the force readings to variables
            lForce_raw, rForce_raw = int(parts[0]), int(parts[1])

            # Get the averaged difference between the FSR data readings in the buffers
            readingTuple = EspReadingsBuffer(lForce_raw, rForce_raw)
            if (readingTuple == -1):
                return -1
            # Return the averaged difference between the FSR data readings
            else:
                return readingTuple
        return -1

"""A function that checks if the users weight distribution is correct during the squat"""
def checkWeightDistribution (right_difference_avg, left_difference_avg):  
    # Initialize weight distribution metrics to 1 (e.g Good)
    # These are changed to 0 if there if a difference in weight distribution
    weightDistributionR = 1
    weightDistributionL = 1
    weightDistributionDifference = 1

    # Initializing a cutOff Value for the pressure sensor readings 
    cutOffValue = 1000
    # If the averaged differences are larger than the cut-off value, change the respective var to 0 ( e.g bad)
    if right_difference_avg > cutOffValue:
        weightDistributionR = 0
    if left_difference_avg > cutOffValue:
        weightDistributionL = 0 
    
    # If there is a large difference between the values, set the var representing left to right weight distribution to 0
    if abs(left_difference_avg - right_difference_avg) > cutOffValue:
        weightDistributionDifference = 0

    # Return a list with the evaluated weight measures (Any 0's returned indicated a weight distribution error)
    return [weightDistributionR, weightDistributionL, weightDistributionDifference]

"""Handles logic for manual data labeling and recording."""
def run_recording_mode(img, tracker, collector_obj, data):
    # Unpack the data into their respective variables
    lean, asym, r_angle, l_angle, avg_angle, weights, speed = data
    
    # Calculate warnings
    active_warnings = []
    if lean > 10: active_warnings.append("Chest UP!")
    if asym > 8: active_warnings.append("Uneven Squat!")
    if avg_angle >= 95: active_warnings.append("Go Lower")
    if 0 in weights: active_warnings.append("Check Weight!")

    # Save Snapshot
    snapshot = [round(lean, 2), round(asym, 2), round(r_angle, 2), round(l_angle, 2), 
                weights[0], weights[1], weights[2], "TRAINING_DATA", round(speed, 2)]
    collector_obj.collect(snapshot)

    # Draw the skeleton and add the text to the camera frame
    img = tracker.draw_skeleton(img, line_color=(0, 0, 255))
    cv2.putText(img, "RECORDING MODE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    y_off = 100
    for w in active_warnings:
        cv2.putText(img, w, (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        y_off += 50
    # Returning the image over-lay to be applied
    return img

""""A function that provides feedback to the user based on the gathered data."""
"""Handles AI prediction, state stabilization, and coaching UI."""
def run_ai_coaching_mode(img, tracker, ai_model, coach, history, data, dims):
    # Unpack the data into their respective variables
    lean, asym, r_angle, l_angle, _, weights, speed = data

    # Unpack the dimensions data
    img_h, img_w = dims

    # Create and group the data frame with the corresponding data that was parsed
    feat_cols = ['lean', 'asymmetry', 'right_angle', 'left_angle', 'force_right', 'force_left', 'force_diff', 'angle_diff']
    features = pd.DataFrame([[lean, asym, r_angle, l_angle, weights[0], weights[1], weights[2], speed]], columns=feat_cols)
    
    # Get the raw prediction from the model
    raw_pred = ai_model.predict(features)[0]

    # Append the prediction the history list
    history.append(raw_pred)

    # Get the stabilized prediction label, by looking at the prediction with the most counts (at this current time)
    stable_pred = max(set(history), key=history.count)

    # Implement Coach logic, using the frames as input
    coach.process_frame(stable_pred, raw_pred)

    # Dynamic UI Coloring, if the user gets to the bottom of the squat, use green as the overlay, otherwise use red
    status_color = (0, 255, 0) if stable_pred == "At_Bottom" else (255, 255, 255)
    if "FIX" in coach.live_feedback: status_color = (0, 0, 255)
    
    # Call the function to draw the skeleton overlay on the image
    img = tracker.draw_skeleton(img, line_color=status_color)
    
    # Provide the on screen  live feedback and rep count
    cv2.putText(img, coach.live_feedback.upper(), (40, 100), cv2.FONT_HERSHEY_DUPLEX, 1.8, status_color, 4)
    cv2.putText(img, f"REPS: {coach.rep_count}", (40, img_h - 60), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 0), 5)
    
    # Provide the verdict on the last reps form
    if coach.last_verdict:
        res_color = (0, 255, 0) if "PERFECT" in coach.last_verdict else (255, 0, 0)
        cv2.putText(img, f"LAST: {coach.last_verdict}", (img_w - 850, img_h - 180), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, res_color, 4)
    return img

"""A function that draws the dynamic progress bar."""
def draw_depth_gauge(img, avg_angle, img_w):
    # If the user is near the bottom, use orange, if they are at the bottom use green, if they are at the top use red
    if avg_angle > 135: bar_color = (0, 0, 255)
    elif avg_angle > 95: bar_color = (0, 165, 255)
    else: bar_color = (0, 255, 0)

    # Defining the box dimensions
    bx, bt, bb = img_w - 100, 150, 450
    
    # Filling the box to a certain level depending on where the user is in the squat
    fill_y = int(np.interp(avg_angle, [95, 180], [bt, bb]))

    # Drawing the grey empty slot
    cv2.rectangle(img, (bx, bt), (bx + 50, bb), (40, 40, 40), -1)

    # Drawing the colored bar
    cv2.rectangle(img, (bx, fill_y), (bx + 50, bb), bar_color, -1)

    # Drawing a white line at the top of the box
    cv2.line(img, (bx - 10, bt), (bx + 60, bt), (255, 255, 255), 3)


    # Outputting the Goal text
    cv2.putText(img, "GOAL", (bx - 15, bt - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

    # Outputting the live degree count
    cv2.putText(img, f"{int(avg_angle)} deg", (bx - 30, bb + 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Returning the image
    return img

"""" A function to calculate the landmarks and angles of the users form """
def analyze_squat_form(img, collector_obj):
    # Access past frame data through global variables
    global LAST_WEIGHT_STATE, LAST_AVG_ANGLE

    # Set up the frame 
    img_h, img_w, _ = img.shape
    
    # Process the frame
    img = tracker.process_frame(img)

    # Extract the landmark points from the video stream (x, y, z) for each landmark specified in the for loop
    landmarks = [tracker.get_landmark_coords(i) for i in [RIGHT_HIP, LEFT_HIP, RIGHT_KNEE, LEFT_KNEE, RIGHT_ANKLE, LEFT_ANKLE, RIGHT_SHOULDER, LEFT_SHOULDER]]

    # Check if all vectors were received, if so, continue
    if all(v is not None for v in landmarks):
        
        # Get the angle between hip, knee, and ankle (in both legs) for squat depth check
        r_angle = tracker.calculate_angle(landmarks[0], landmarks[2], landmarks[4])
        l_angle = tracker.calculate_angle(landmarks[1], landmarks[3], landmarks[5])

        # Get the average angle between the two legs
        avg_angle = (r_angle + l_angle) / 2

        # Get the difference between the two angles
        asym = abs(r_angle - l_angle)

        # Getting angle from right shoulder to right hip and left shoulder to left hip, and averaging them to get the lean
        lean = (tracker.calculate_torso_lean(landmarks[6], landmarks[0], img_w, img_h) + 
                tracker.calculate_torso_lean(landmarks[7], landmarks[1], img_w, img_h)) / 2
        
        # Get the difference between the last average angle between the two legs and the current, as a way to log motion changes
        speed = avg_angle - LAST_AVG_ANGLE
        LAST_AVG_ANGLE = avg_angle

        # Read a fsr reading packet from the ESP32 micro-controller
        sensor_data = esp.read_packet()

        # Calling a function to evaluate the sensor data, saving the result
        forces = EvaluateForceReadings(sensor_data)

        # If forces return a tuple, call the checkWeightDistribution function, and pass the values in the tuple
        if forces != -1 and forces is not None:
            # Get the weight distribution status
            LAST_WEIGHT_STATE = checkWeightDistribution(forces[0], forces[1])

        # Combine all data points into a tuple
        current_data = (lean, asym, r_angle, l_angle, avg_angle, LAST_WEIGHT_STATE, speed)

        # If the recording option is selected, run the live feedback mode (For data collection)
        if collector_obj.is_recording:
            img = run_recording_mode(img, tracker, collector_obj, current_data)
        # Else run the real ai coaching mode
        else:
            img = run_ai_coaching_mode(img, tracker, ai_model, coach, prediction_history, current_data, (img_h, img_w))
            
            # Call the function too provide a on screen depth gauge to the user
            img = draw_depth_gauge(img, avg_angle, img_w)

    return img

# -- Main Execution Loop --
while True:
    # Get the video stream
    success, img = stream.read_frame()

    # Re-run the loop if the video stream is not gotten
    if not success: continue

    # Listen for key strokes
    key = cv2.waitKey(1) & 0xFF

    # If the "r" key is clicked, toggle the recording functionality
    if key == ord('r'):
            if not collector.is_recording:
                collector.start()
            else:
                collector.stop()

    # Process the frame logic
    img = analyze_squat_form(img, collector)

    # Show the image (From the webcam) on the screen
    cv2.imshow("Squat Form Tracker", img)

    # If the q button is clicked, stop the program
    if key == ord('q'):
        break

# Cleanup the objects
esp.stop()
stream.stop()
cv2.destroyAllWindows()