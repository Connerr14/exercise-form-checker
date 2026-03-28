import cv2

class VideoStream:
    def __init__(self, camera_index=0):
        """Initializes the webcam."""
        self.cap = cv2.VideoCapture(camera_index)

    def read_frame(self):
        """Grabs a frame. Returns a boolean for success, and the image."""
        success, img = self.cap.read()
        return success, img

    def stop(self):
        """Safely releases the camera."""
        self.cap.release()