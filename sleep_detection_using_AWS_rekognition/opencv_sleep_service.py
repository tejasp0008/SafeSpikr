import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
import math
from sleep_config import SleepDetectionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenCVFacialLandmarks:
    """OpenCV-based facial landmarks container"""
    def __init__(self):
        self.left_eye = []
        self.right_eye = []
        self.nose = []
        self.mouth = []
        self.face_rect = None
        self.landmarks_68 = None  # 68-point facial landmarks if available

class OpenCVEmotionData:
    """Basic emotion inference from facial features"""
    def __init__(self):
        self.emotions = {
            'CALM': 0.0,
            'SURPRISED': 0.0,
            'CONFUSED': 0.0,
            'ANGRY': 0.0,
            'DISGUSTED': 0.0,
            'FEAR': 0.0,
            'HAPPY': 0.0,
            'SAD': 0.0
        }
        self.dominant_emotion = 'CALM'
        self.confidence = 0.0

class HeadMovement:
    """Container for head movement analysis"""
    def __init__(self):
        self.angle_change = 0.0
        self.movement_magnitude = 0.0
        self.direction = 'STABLE'  # STABLE, LEFT, RIGHT, UP, DOWN

class OpenCVSleepService:
    """OpenCV-based fallback service for sleep detection"""
    
    def __init__(self):
        self.config = SleepDetectionConfig()
        self.face_cascade = None
        self.eye_cascade = None
        self.is_available = False
        self.last_error = None
        self.previous_face_center = None
        self._initialize_cascades()
    
    def _initialize_cascades(self):
        """Initialize OpenCV cascade classifiers"""
        try:
            # Load face cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Load eye cascade
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Verify cascades loaded successfully
            if self.face_cascade.empty() or self.eye_cascade.empty():
                raise Exception("Failed to load cascade classifiers")
            
            self.is_available = True
            logger.info("OpenCV cascade classifiers loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV cascades: {e}")
            self.last_error = str(e)
            self.is_available = False
    
    def detect_face_landmarks(self, frame: np.ndarray) -> Optional[OpenCVFacialLandmarks]:
        """Detect facial landmarks using OpenCV with improved closed eye detection"""
        if not self.is_available:
            return None
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with multiple attempts for better detection when eyes are closed
            faces = []
            
            # Try different detection parameters
            detection_params = [
                # More sensitive detection
                {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)},
                # Standard detection
                {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                # Less strict detection
                {'scaleFactor': 1.2, 'minNeighbors': 2, 'minSize': (25, 25)},
            ]
            
            for params in detection_params:
                faces = self.face_cascade.detectMultiScale(gray, **params)
                if len(faces) > 0:
                    break
            
            # If still no faces, try with profile face cascade or alternative method
            if len(faces) == 0:
                # Enhance image contrast for better detection
                enhanced = cv2.equalizeHist(gray)
                faces = self.face_cascade.detectMultiScale(
                    enhanced, 
                    scaleFactor=1.05, 
                    minNeighbors=2, 
                    minSize=(20, 20)
                )
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = face
            
            landmarks = OpenCVFacialLandmarks()
            landmarks.face_rect = (x, y, w, h)
            
            # Use a different approach for eye detection - analyze the eye regions directly
            eye_regions = self._analyze_eye_regions(gray, x, y, w, h)
            
            if eye_regions:
                landmarks.left_eye = eye_regions['left_eye']
                landmarks.right_eye = eye_regions['right_eye']
            
            # Estimate nose position (center-bottom of face)
            nose_x = x + w // 2
            nose_y = y + int(h * 0.6)
            landmarks.nose = [(nose_x / frame.shape[1], nose_y / frame.shape[0])]
            
            # Estimate mouth position (center-bottom of face)
            mouth_x = x + w // 2
            mouth_y = y + int(h * 0.8)
            landmarks.mouth = [(mouth_x / frame.shape[1], mouth_y / frame.shape[0])]
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error detecting face landmarks: {e}")
            self.last_error = str(e)
            return None
    
    def _analyze_eye_regions(self, gray_image, face_x, face_y, face_w, face_h):
        """Analyze eye regions directly to detect open/closed state"""
        try:
            # Define eye regions based on typical face proportions
            left_eye_region = {
                'x': face_x + int(face_w * 0.15),
                'y': face_y + int(face_h * 0.25),
                'w': int(face_w * 0.25),
                'h': int(face_h * 0.15)
            }
            
            right_eye_region = {
                'x': face_x + int(face_w * 0.60),
                'y': face_y + int(face_h * 0.25),
                'w': int(face_w * 0.25),
                'h': int(face_h * 0.15)
            }
            
            # Extract eye regions
            left_roi = gray_image[left_eye_region['y']:left_eye_region['y']+left_eye_region['h'],
                                 left_eye_region['x']:left_eye_region['x']+left_eye_region['w']]
            
            right_roi = gray_image[right_eye_region['y']:right_eye_region['y']+right_eye_region['h'],
                                  right_eye_region['x']:right_eye_region['x']+right_eye_region['w']]
            
            # Analyze each eye region
            left_eye_landmarks = self._analyze_single_eye_region(left_roi, left_eye_region, gray_image.shape)
            right_eye_landmarks = self._analyze_single_eye_region(right_roi, right_eye_region, gray_image.shape)
            
            return {
                'left_eye': left_eye_landmarks,
                'right_eye': right_eye_landmarks
            }
            
        except Exception as e:
            logger.error(f"Error analyzing eye regions: {e}")
            return None
    
    def _analyze_single_eye_region(self, eye_roi, region_info, frame_shape):
        """Analyze a single eye region to determine if it's open or closed"""
        try:
            if eye_roi.size == 0:
                return None
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(eye_roi, (3, 3), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate the aspect ratio of the eye region
            region_aspect_ratio = region_info['w'] / region_info['h'] if region_info['h'] > 0 else 1.0
            
            # Analyze the intensity distribution to detect closed eyes
            mean_intensity = cv2.mean(eye_roi)[0]
            
            # Calculate variance to detect uniformity (closed eyes tend to be more uniform)
            variance = cv2.meanStdDev(eye_roi)[1][0][0]
            
            # Determine if eye is likely closed based on multiple factors
            is_likely_closed = (
                variance < 20 or  # Low variance indicates uniform region (closed eye)
                mean_intensity > 180 or  # Very bright (possibly closed eye)
                len(contours) < 2  # Few contours (closed eye)
            )
            
            # Create eye landmarks based on analysis
            frame_height, frame_width = frame_shape
            
            # Convert region coordinates to normalized coordinates
            center_x = (region_info['x'] + region_info['w'] // 2) / frame_width
            center_y = (region_info['y'] + region_info['h'] // 2) / frame_height
            
            if is_likely_closed:
                # Create flatter eye shape for closed eyes
                eye_width = region_info['w'] / frame_width
                eye_height = (region_info['h'] * 0.3) / frame_height  # Much smaller height
                
                eye_landmarks = [
                    (center_x - eye_width/2, center_y),           # Left corner
                    (center_x - eye_width/4, center_y - eye_height/4),  # Top left
                    (center_x, center_y - eye_height/4),          # Top center
                    (center_x + eye_width/4, center_y - eye_height/4),  # Top right
                    (center_x + eye_width/2, center_y),           # Right corner
                    (center_x, center_y + eye_height/4)           # Bottom center
                ]
            else:
                # Create normal eye shape for open eyes
                eye_width = region_info['w'] / frame_width
                eye_height = region_info['h'] / frame_height
                
                eye_landmarks = [
                    (center_x - eye_width/2, center_y),           # Left corner
                    (center_x - eye_width/4, center_y - eye_height/2),  # Top left
                    (center_x, center_y - eye_height/2),          # Top center
                    (center_x + eye_width/4, center_y - eye_height/2),  # Top right
                    (center_x + eye_width/2, center_y),           # Right corner
                    (center_x, center_y + eye_height/2)           # Bottom center
                ]
            
            return eye_landmarks
            
        except Exception as e:
            logger.error(f"Error analyzing single eye region: {e}")
            return None
    
    def _create_eye_landmarks(self, eye_rect: Tuple[int, int, int, int], face_x: int, face_y: int, frame_width: int = 640, frame_height: int = 480) -> List[Tuple[float, float]]:
        """Create eye landmark points from eye rectangle"""
        ex, ey, ew, eh = eye_rect
        
        # Convert to absolute coordinates
        abs_x = face_x + ex
        abs_y = face_y + ey
        
        # For closed eyes, the height (eh) will be very small
        # Adjust the eye shape based on the detected height
        if eh < ew * 0.3:  # Likely closed eye (height < 30% of width)
            # Create flatter eye shape for closed eyes
            eye_points = [
                (abs_x, abs_y + eh//2),           # Left corner
                (abs_x + ew//4, abs_y + eh//4),   # Top left (closer to center)
                (abs_x + ew//2, abs_y + eh//4),   # Top center (closer to center)
                (abs_x + 3*ew//4, abs_y + eh//4), # Top right (closer to center)
                (abs_x + ew, abs_y + eh//2),      # Right corner
                (abs_x + ew//2, abs_y + 3*eh//4)  # Bottom center (closer to center)
            ]
        else:
            # Normal open eye shape
            eye_points = [
                (abs_x, abs_y + eh//2),      # Left corner
                (abs_x + ew//4, abs_y),      # Top left
                (abs_x + ew//2, abs_y),      # Top center
                (abs_x + 3*ew//4, abs_y),    # Top right
                (abs_x + ew, abs_y + eh//2), # Right corner
                (abs_x + ew//2, abs_y + eh)  # Bottom center
            ]
        
        # Normalize coordinates (0-1 range)
        return [(x/frame_width, y/frame_height) for x, y in eye_points]
    
    def calculate_eye_aspect_ratio(self, eye_points: List[Tuple[float, float]]) -> float:
        """Calculate eye aspect ratio for sleep detection"""
        if not eye_points or len(eye_points) < 6:
            return 0.15  # Lower default value for closed eyes
        
        try:
            # Calculate vertical distances (eye height)
            # Points: [left_corner, top_left, top_center, top_right, right_corner, bottom_center]
            vertical_1 = abs(eye_points[1][1] - eye_points[5][1])  # top_left to bottom_center
            vertical_2 = abs(eye_points[2][1] - eye_points[5][1])  # top_center to bottom_center
            vertical_3 = abs(eye_points[3][1] - eye_points[5][1])  # top_right to bottom_center
            
            # Calculate horizontal distance (eye width)
            horizontal = abs(eye_points[0][0] - eye_points[4][0])  # left_corner to right_corner
            
            # Calculate aspect ratio using average of vertical distances
            if horizontal > 0:
                avg_vertical = (vertical_1 + vertical_2 + vertical_3) / 3.0
                ear = avg_vertical / horizontal
                
                # Clamp the value to reasonable range
                ear = max(0.05, min(0.8, ear))
                
                return ear
            
            return 0.15  # Closed eye default
            
        except (IndexError, ZeroDivisionError, TypeError):
            return 0.15  # Safe default for closed eyes
    
    def detect_head_movement(self, frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> HeadMovement:
        """Detect head movement between frames"""
        movement = HeadMovement()
        
        if not self.is_available or previous_frame is None:
            return movement
        
        try:
            # Convert frames to grayscale
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in both frames
            faces_current = self.face_cascade.detectMultiScale(gray_current, 1.1, 5, minSize=(30, 30))
            faces_previous = self.face_cascade.detectMultiScale(gray_previous, 1.1, 5, minSize=(30, 30))
            
            if len(faces_current) > 0 and len(faces_previous) > 0:
                # Get largest faces
                face_current = max(faces_current, key=lambda rect: rect[2] * rect[3])
                face_previous = max(faces_previous, key=lambda rect: rect[2] * rect[3])
                
                # Calculate face centers
                center_current = (face_current[0] + face_current[2]//2, face_current[1] + face_current[3]//2)
                center_previous = (face_previous[0] + face_previous[2]//2, face_previous[1] + face_previous[3]//2)
                
                # Calculate movement
                dx = center_current[0] - center_previous[0]
                dy = center_current[1] - center_previous[1]
                
                movement.movement_magnitude = math.sqrt(dx*dx + dy*dy)
                
                # Determine direction
                if movement.movement_magnitude > 10:  # Threshold for significant movement
                    if abs(dx) > abs(dy):
                        movement.direction = 'RIGHT' if dx > 0 else 'LEFT'
                    else:
                        movement.direction = 'DOWN' if dy > 0 else 'UP'
                    
                    # Calculate angle change (simplified)
                    movement.angle_change = math.degrees(math.atan2(dy, dx))
                
                self.previous_face_center = center_current
            
            return movement
            
        except Exception as e:
            logger.error(f"Error detecting head movement: {e}")
            self.last_error = str(e)
            return movement
    
    def infer_basic_emotion(self, landmarks: OpenCVFacialLandmarks, eye_aspect_ratio: float) -> OpenCVEmotionData:
        """Basic emotion inference from facial features"""
        emotion_data = OpenCVEmotionData()
        
        if not landmarks:
            return emotion_data
        
        try:
            # Simple heuristics for emotion detection
            
            # If eyes are nearly closed, likely calm or sleepy
            if eye_aspect_ratio < 0.2:
                emotion_data.emotions['CALM'] = 80.0
                emotion_data.dominant_emotion = 'CALM'
                emotion_data.confidence = 80.0
            
            # If eyes are wide open, might be surprised or alert
            elif eye_aspect_ratio > 0.4:
                emotion_data.emotions['SURPRISED'] = 60.0
                emotion_data.dominant_emotion = 'SURPRISED'
                emotion_data.confidence = 60.0
            
            # Normal eye opening
            else:
                emotion_data.emotions['CALM'] = 70.0
                emotion_data.dominant_emotion = 'CALM'
                emotion_data.confidence = 70.0
            
            return emotion_data
            
        except Exception as e:
            logger.error(f"Error inferring emotion: {e}")
            return emotion_data
    
    def analyze_frame_opencv(self, frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Complete frame analysis using OpenCV"""
        if not self.is_available:
            return {'success': False, 'error': 'OpenCV service not available'}
        
        try:
            # Detect facial landmarks
            landmarks = self.detect_face_landmarks(frame)
            
            if not landmarks:
                return {'success': False, 'error': 'No face detected'}
            
            # Calculate eye aspect ratios
            left_ear = self.calculate_eye_aspect_ratio(landmarks.left_eye) if landmarks.left_eye else 0.3
            right_ear = self.calculate_eye_aspect_ratio(landmarks.right_eye) if landmarks.right_eye else 0.3
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Detect head movement
            head_movement = self.detect_head_movement(frame, previous_frame)
            
            # Infer basic emotion
            emotion_data = self.infer_basic_emotion(landmarks, avg_ear)
            
            return {
                'success': True,
                'landmarks': landmarks,
                'eye_aspect_ratio': avg_ear,
                'head_movement': head_movement,
                'emotion_data': emotion_data,
                'face_detected': True,
                'confidence': 75.0  # Fixed confidence for OpenCV detection
            }
            
        except Exception as e:
            logger.error(f"Error in OpenCV frame analysis: {e}")
            self.last_error = str(e)
            return {'success': False, 'error': str(e)}
    
    def is_service_available(self) -> bool:
        """Check if OpenCV service is available"""
        return self.is_available
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message"""
        return self.last_error
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            'available': self.is_available,
            'last_error': self.last_error,
            'face_cascade_loaded': self.face_cascade is not None and not self.face_cascade.empty(),
            'eye_cascade_loaded': self.eye_cascade is not None and not self.eye_cascade.empty()
        }