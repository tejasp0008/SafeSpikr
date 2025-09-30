import cv2
import numpy as np
from PIL import Image
import io
import threading
import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import logging
from sleep_config import SleepDetectionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameProcessor:
    """Handles frame preprocessing for optimal analysis"""
    
    def __init__(self):
        self.target_width = 640
        self.target_height = 480
        self.quality = 85
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal AWS Rekognition analysis"""
        if frame is None:
            return None
        
        # Resize frame to target resolution for consistent processing
        height, width = frame.shape[:2]
        if width != self.target_width or height != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        # Enhance contrast and brightness for better face detection
        frame = self._enhance_image_quality(frame)
        
        return frame
    
    def _enhance_image_quality(self, frame: np.ndarray) -> np.ndarray:
        """Enhance image quality for better detection"""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """Convert OpenCV frame to bytes for AWS Rekognition"""
        if frame is None:
            return b''
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to bytes with JPEG compression
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=self.quality)
            
            return img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting frame to bytes: {e}")
            return b''
    
    def add_detection_overlay(self, frame: np.ndarray, detection_result: Dict[str, Any], 
                             landmarks: Any = None, system_info: Dict[str, Any] = None) -> np.ndarray:
        """Add comprehensive detection overlays to frame"""
        if frame is None:
            return frame
        
        try:
            # Import visual overlay manager
            from visual_overlay_manager import VisualOverlayManager
            from sleep_detection_engine import DetectionResult, SleepMetrics
            
            # Initialize overlay manager if not exists
            if not hasattr(self, '_overlay_manager'):
                self._overlay_manager = VisualOverlayManager()
            
            # Convert dict to DetectionResult if needed
            if isinstance(detection_result, dict):
                # Extract metrics
                metrics_data = detection_result.get('metrics', {})
                if isinstance(metrics_data, dict):
                    metrics = SleepMetrics(
                        eye_closure_duration=metrics_data.get('eye_closure_duration', 0.0),
                        blink_rate=metrics_data.get('blink_rate', 0.0),
                        head_movement_angle=metrics_data.get('head_movement_angle', 0.0),
                        drowsiness_score=metrics_data.get('drowsiness_score', 0.0),
                        distraction_score=metrics_data.get('distraction_score', 0.0),
                        attention_score=metrics_data.get('attention_score', 100.0),
                        eye_aspect_ratio=metrics_data.get('eye_aspect_ratio', 0.3),
                        head_stability=metrics_data.get('head_stability', 1.0)
                    )
                else:
                    metrics = metrics_data
                
                # Create DetectionResult
                result = DetectionResult(
                    state=detection_result.get('state', 'unknown'),
                    confidence=detection_result.get('confidence', 0.0),
                    metrics=metrics,
                    timestamp=datetime.now()
                )
            else:
                result = detection_result
            
            # Create comprehensive overlay
            overlay_frame = self._overlay_manager.create_comprehensive_overlay(
                frame, result, landmarks, system_info
            )
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Error adding detection overlay: {e}")
            # Fallback to simple overlay
            return self._add_simple_overlay(frame, detection_result)
    
    def _add_simple_overlay(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """Simple fallback overlay"""
        overlay_frame = frame.copy()
        
        try:
            # Add basic state indicator
            state = detection_result.get('state', 'unknown')
            confidence = detection_result.get('confidence', 0.0)
            
            # Choose color based on state
            color_map = {
                'normal': (0, 255, 0),      # Green
                'drowsy': (0, 255, 255),    # Yellow
                'sleeping': (0, 0, 255),    # Red
                'distracted': (255, 0, 255) # Magenta
            }
            color = color_map.get(state, (128, 128, 128))
            
            # Add state text
            state_text = f"{state.upper()}: {confidence:.1f}%"
            cv2.putText(overlay_frame, state_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(overlay_frame, timestamp, (overlay_frame.shape[1] - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Error adding simple overlay: {e}")
            return frame

class CameraManager:
    """Independent camera management for sleep detection"""
    
    def __init__(self):
        self.config = SleepDetectionConfig()
        self.camera_index = self.config.CAMERA_INDEX
        self.cap = None
        self.is_running = False
        self.frame_processor = FrameProcessor()
        
        # Frame management
        self.current_frame = None
        self.previous_frame = None
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Threading
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        
        # Callbacks
        self.frame_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'dropped_frames': 0,
            'average_fps': 0.0,
            'last_frame_time': None
        }
        
        logger.info(f"Camera Manager initialized for camera index {self.camera_index}")
    
    def start_camera(self) -> bool:
        """Initialize and start the camera"""
        try:
            # Release any existing camera
            if self.cap is not None:
                self.cap.release()
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("Camera test capture failed")
                return False
            
            logger.info(f"Camera {self.camera_index} started successfully")
            logger.info(f"Camera resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def start_capture_thread(self, frame_callback: Optional[Callable] = None):
        """Start threaded frame capture"""
        if self.is_running:
            logger.warning("Capture thread already running")
            return
        
        if not self.cap or not self.cap.isOpened():
            if not self.start_camera():
                logger.error("Cannot start capture thread - camera not available")
                return
        
        self.frame_callback = frame_callback
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Camera capture thread started")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        frame_skip_counter = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    self.stats['dropped_frames'] += 1
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                
                self.stats['total_frames'] += 1
                
                # Frame skipping for performance (process every Nth frame)
                frame_skip_counter += 1
                if frame_skip_counter < self.config.FRAME_SKIP_RATE:
                    continue
                
                frame_skip_counter = 0
                
                # Preprocess frame
                processed_frame = self.frame_processor.preprocess_frame(frame)
                
                # Update frame data with thread safety
                with self.frame_lock:
                    self.previous_frame = self.current_frame
                    self.current_frame = processed_frame
                    self.frame_count += 1
                    self.stats['processed_frames'] += 1
                    self.stats['last_frame_time'] = datetime.now()
                
                # Calculate FPS
                self._update_fps()
                
                # Call frame callback if provided
                if self.frame_callback and processed_frame is not None:
                    try:
                        self.frame_callback(processed_frame, self.previous_frame)
                    except Exception as e:
                        logger.error(f"Error in frame callback: {e}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Longer pause on error
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.stats['average_fps'] = self.fps
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame (synchronous)"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return self.frame_processor.preprocess_frame(frame)
            return None
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame from threaded capture"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_previous_frame(self) -> Optional[np.ndarray]:
        """Get the previous frame for motion analysis"""
        with self.frame_lock:
            return self.previous_frame.copy() if self.previous_frame is not None else None
    
    def get_frame_pair(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current and previous frames as a pair"""
        with self.frame_lock:
            current = self.current_frame.copy() if self.current_frame is not None else None
            previous = self.previous_frame.copy() if self.previous_frame is not None else None
            return current, previous
    
    def frame_to_bytes(self, frame: Optional[np.ndarray] = None) -> bytes:
        """Convert frame to bytes for AWS processing"""
        if frame is None:
            frame = self.get_current_frame()
        
        if frame is None:
            return b''
        
        return self.frame_processor.frame_to_bytes(frame)
    
    def add_overlay(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """Add detection overlay to frame"""
        return self.frame_processor.add_detection_overlay(frame, detection_result)
    
    def stop_capture(self):
        """Stop the capture thread"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            logger.info("Capture thread stopped")
    
    def release_camera(self):
        """Release camera resources"""
        self.stop_capture()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        logger.info("Camera resources released")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and status"""
        if not self.cap:
            return {'available': False, 'error': 'Camera not initialized'}
        
        try:
            return {
                'available': self.cap.isOpened(),
                'index': self.camera_index,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'current_fps': self.fps,
                'is_running': self.is_running,
                'frame_count': self.stats['processed_frames']
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return {
            **self.stats,
            'current_fps': self.fps,
            'is_running': self.is_running,
            'camera_available': self.cap is not None and self.cap.isOpened()
        }
    
    def reset_statistics(self):
        """Reset capture statistics"""
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'dropped_frames': 0,
            'average_fps': 0.0,
            'last_frame_time': None
        }
        self.frame_count = 0
        logger.info("Camera statistics reset")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.release_camera()