import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional
from config import Config

class CameraManager:
    def __init__(self):
        self.camera_index = Config.CAMERA_INDEX
        self.cap = None
    
    def start_camera(self) -> bool:
        """Initialize and start the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """Convert OpenCV frame to bytes for AWS Rekognition"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()
    
    def display_frame(self, frame: np.ndarray, window_name: str = "Camera Feed"):
        """Display frame in a window"""
        cv2.imshow(window_name, frame)
    
    def release_camera(self):
        """Release the camera resource"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()