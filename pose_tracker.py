import cv2
import mediapipe as mp
import numpy as np

class PoseTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence, 
                min_tracking_confidence=min_tracking_confidence
            )
            self.results = None

    def process_frame(self, img):
        """Processes the image and finds the pose, but DOES NOT draw it yet."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        return img

    def draw_skeleton(self, img, line_color=(255, 255, 255), dot_color=(0, 0, 255)):
        """Draws the skeleton using custom BGR colors."""
        if self.results and self.results.pose_landmarks:
            # Setup custom drawing specs based on the colors passed in
            connection_spec = self.mp_drawing.DrawingSpec(color=line_color, thickness=3, circle_radius=2)
            landmark_spec = self.mp_drawing.DrawingSpec(color=dot_color, thickness=3, circle_radius=3)
            
            self.mp_drawing.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )
        return img
    
    def get_landmark_coords(self, landmark_index):
        """Safely extracts the [x, y, z] coordinates of a specific joint."""
        if not self.results or not self.results.pose_landmarks:
            return None
        
        landmark = self.results.pose_landmarks.landmark[landmark_index]
        return np.array([landmark.x, landmark.y, landmark.z])

    def calculate_angle(self, a, b, c):
        """
        Calculates the angle between three points (a, b, c).
        'b' is the vertex (e.g., the knee).
        """
        if a is None or b is None or c is None:
            return None
            
        # Create vectors from the vertex (b) to the endpoints (a and c)
        ba = a - b
        bc = c - b
        
        # Calculate the cosine of the angle using the dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Handle potential floating-point errors outside the valid domain of arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    
    def calculate_torso_lean(self, shoulder, hip, img_width, img_height):
        """Calculates the 2D angle of the torso from a perfect vertical line."""
        if shoulder is None or hip is None:
            return None
            
        # Convert normalized coordinates to actual pixel coordinates
        shoulder_x = shoulder[0] * img_width
        shoulder_y = shoulder[1] * img_height
        hip_x = hip[0] * img_width
        hip_y = hip[1] * img_height
        
        # Calculate 2D vector differences
        dx = shoulder_x - hip_x
        # OpenCV Y-axis goes top-to-bottom. Hip Y will be larger than Shoulder Y.
        dy = hip_y - shoulder_y 
        
        # Calculate angle from vertical using arctangent
        # Using abs(dx) ensures it works whether you face left or right
        angle = np.degrees(np.arctan2(abs(dx), dy))
        return angle