import cv2
import mediapipe as mp
import numpy as np

class PoseTracker:
    # This constructor sets up the mediapipe tooling
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
            # Initializing the pose feature
            self.mp_pose = mp.solutions.pose

            # Initializing the drawing feature
            self.mp_drawing = mp.solutions.drawing_utils

            # Setting up the confidence levels
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence, 
                min_tracking_confidence=min_tracking_confidence
            )
            self.results = None

    """Processes the image and finds the pose"""
    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        return img

    """This function draws the skeleton on the camera frame using custom BGR colors."""
    def draw_skeleton(self, img, line_color=(255, 255, 255), dot_color=(0, 0, 255)):
        # Check if there is results
        if self.results and self.results.pose_landmarks:
            # Setup custom drawing specs based on the colors passed in
            connection_spec = self.mp_drawing.DrawingSpec(color=line_color, thickness=3, circle_radius=2)
            landmark_spec = self.mp_drawing.DrawingSpec(color=dot_color, thickness=3, circle_radius=3)
            
            # Configure the drawings on the landmarks
            self.mp_drawing.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )
        # Return overlaying drawings
        return img
    
    """Extracts the [x, y, z] coordinates of a specific joint."""
    def get_landmark_coords(self, landmark_index):
        # Exit if there are is nothing to extract
        if not self.results or not self.results.pose_landmarks:
            return None
        
        # Process x, y, and z for a specific joint, and return the array
        landmark = self.results.pose_landmarks.landmark[landmark_index]
        return np.array([landmark.x, landmark.y, landmark.z])

    """
        Calculates the angle between three points (a, b, c) using the dot product.
    """
    def calculate_angle(self, a, b, c):
        # Return if one of the angles is missing
        if a is None or b is None or c is None:
            return None
            
        # Create vectors from the vertex (b) to the endpoints (a and c)
        ba = a - b
        bc = c - b
        
        # Calculate the cosine of the angle using the dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Handle potential floating-point errors outside the valid domain
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.degrees(np.arccos(cosine_angle))

        # Returns the angle in degrees
        return angle
    
    """Calculates the 2D angle of the torso from a perfect vertical line."""
    def calculate_torso_lean(self, shoulder, hip, img_width, img_height):

       # Return if the shoulder or hip metric is missing
        if shoulder is None or hip is None:
            return None
            
        # Convert normalized coordinates to actual pixel coordinates, for processing and displaying
        shoulder_x = shoulder[0] * img_width
        shoulder_y = shoulder[1] * img_height
        hip_x = hip[0] * img_width
        hip_y = hip[1] * img_height
        
        # Calculate 2D vector differences (If the shoulder is directly above the hip, the dx is 0, otherwise it changes)
        dx = shoulder_x - hip_x

        # Getting the vertical height (hip up to shoulder). 
        # OpenCV Y-axis goes top-to-bottom so Hip Y will be larger than Shoulder Y.
        dy = hip_y - shoulder_y 
        
        # Calculate angle from vertical using arctangent
        # Using abs(dx) ensures it works whether you face left or right
        angle = np.degrees(np.arctan2(abs(dx), dy))
        return angle