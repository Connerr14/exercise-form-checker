import cv2
# This class is for camera manipulation
class VideoStream:
    # Initializing the camera with cv2 (OpenCV)
    def __init__(self, camera_index=0):
        """Initializes the webcam."""
        self.cap = cv2.VideoCapture(camera_index)

    # A function for getting the camera frame
    def read_frame(self):
        """Grabs a frame. Returns a boolean for success, and the image."""
        success, img = self.cap.read()
        return success, img

    # A function to stop the camera function
    def stop(self):
        """Releases the camera."""
        self.cap.release()