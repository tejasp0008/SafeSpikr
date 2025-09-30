import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, List, Dict, Any, Tuple
import time
import logging
from sleep_config import SleepDetectionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialLandmarks:
    """Container for facial landmark coordinates"""
    def __init__(self, landmarks_data: Dict[str, Any]):
        self.left_eye = self._extract_eye_points(landmarks_data, 'leftEye')
        self.right_eye = self._extract_eye_points(landmarks_data, 'rightEye')
        self.nose = self._extract_feature_points(landmarks_data, 'nose')
        self.mouth = self._extract_feature_points(landmarks_data, 'mouth')
        self.left_eyebrow = self._extract_feature_points(landmarks_data, 'leftEyeBrow')
        self.right_eyebrow = self._extract_feature_points(landmarks_data, 'rightEyeBrow')
    
    def _extract_eye_points(self, landmarks: Dict, eye_key: str) -> List[Tuple[float, float]]:
        """Extract eye landmark points"""
        if eye_key in landmarks:
            return [(point['X'], point['Y']) for point in landmarks[eye_key]]
        return []
    
    def _extract_feature_points(self, landmarks: Dict, feature_key: str) -> List[Tuple[float, float]]:
        """Extract general feature landmark points"""
        if feature_key in landmarks:
            return [(point['X'], point['Y']) for point in landmarks[feature_key]]
        return []

class EmotionData:
    """Container for emotion analysis data"""
    def __init__(self, emotions_list: List[Dict[str, Any]]):
        self.emotions = {}
        self.dominant_emotion = None
        self.confidence = 0.0
        
        if emotions_list:
            # Convert emotions list to dictionary
            for emotion in emotions_list:
                self.emotions[emotion['Type']] = emotion['Confidence']
            
            # Find dominant emotion
            self.dominant_emotion = max(self.emotions, key=self.emotions.get)
            self.confidence = self.emotions[self.dominant_emotion]

class FacialAnalysis:
    """Container for complete facial analysis results"""
    def __init__(self, face_details: Dict[str, Any]):
        self.landmarks = None
        self.emotions = None
        self.quality = 0.0
        self.confidence = 0.0
        self.pose = {}
        self.eye_states = {}
        
        if face_details:
            # Extract landmarks
            if 'Landmarks' in face_details:
                landmarks_dict = {}
                for landmark in face_details['Landmarks']:
                    landmark_type = landmark['Type']
                    if landmark_type not in landmarks_dict:
                        landmarks_dict[landmark_type] = []
                    landmarks_dict[landmark_type].append({
                        'X': landmark['X'],
                        'Y': landmark['Y']
                    })
                self.landmarks = FacialLandmarks(landmarks_dict)
            
            # Extract emotions
            if 'Emotions' in face_details:
                self.emotions = EmotionData(face_details['Emotions'])
            
            # Extract quality and confidence
            if 'Quality' in face_details:
                quality_data = face_details['Quality']
                self.quality = (quality_data.get('Brightness', 0) + quality_data.get('Sharpness', 0)) / 2
            
            if 'Confidence' in face_details:
                self.confidence = face_details['Confidence']
            
            # Extract pose information
            if 'Pose' in face_details:
                self.pose = face_details['Pose']
            
            # Extract eye states
            if 'EyesOpen' in face_details:
                self.eye_states['eyes_open'] = face_details['EyesOpen']
            if 'EyeDirection' in face_details:
                self.eye_states['eye_direction'] = face_details['EyeDirection']

class AWSRekognitionSleepService:
    """AWS Rekognition service for sleep detection analysis"""
    
    def __init__(self):
        self.config = SleepDetectionConfig()
        self.client = None
        self.is_available = False
        self.last_error = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AWS Rekognition client"""
        try:
            credentials = self.config.get_aws_credentials()
            
            if not credentials['aws_access_key_id'] or not credentials['aws_secret_access_key']:
                logger.warning("AWS credentials not configured")
                return
            
            self.client = boto3.client('rekognition', **credentials)
            
            # Test the connection
            self._test_connection()
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            self.last_error = "AWS credentials not configured"
        except Exception as e:
            logger.error(f"Failed to initialize AWS Rekognition client: {e}")
            self.last_error = str(e)
    
    def _test_connection(self):
        """Test AWS Rekognition connection"""
        try:
            # Simple test call to verify connection
            self.client.list_collections(MaxResults=1)
            self.is_available = True
            logger.info("AWS Rekognition service is available")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['AccessDenied', 'UnauthorizedOperation']:
                logger.error("AWS Rekognition access denied - check permissions")
                self.last_error = "AWS access denied - check permissions"
            else:
                logger.error(f"AWS Rekognition connection test failed: {e}")
                self.last_error = f"Connection test failed: {error_code}"
        except Exception as e:
            logger.error(f"Unexpected error testing AWS connection: {e}")
            self.last_error = str(e)
    
    def analyze_facial_features(self, image_bytes: bytes) -> Optional[FacialAnalysis]:
        """Analyze facial features for sleep detection"""
        if not self.is_available or not self.client:
            return None
        
        try:
            response = self.client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']  # Get all facial attributes including emotions, pose, etc.
            )
            
            face_details = response.get('FaceDetails', [])
            
            if face_details:
                # Return analysis for the first (most prominent) face
                return FacialAnalysis(face_details[0])
            
            return None
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'InvalidParameterException':
                # Usually means no face detected or poor image quality
                logger.debug("No face detected in image or poor image quality")
                return None
            elif error_code == 'ThrottlingException':
                logger.warning("AWS API rate limit exceeded")
                time.sleep(1)  # Brief pause for rate limiting
                return None
            else:
                logger.error(f"AWS Rekognition error: {e}")
                self.last_error = f"API Error: {error_code}"
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in facial analysis: {e}")
            self.last_error = str(e)
            return None
    
    def detect_emotions(self, image_bytes: bytes) -> Optional[EmotionData]:
        """Detect emotions in the image"""
        analysis = self.analyze_facial_features(image_bytes)
        return analysis.emotions if analysis else None
    
    def extract_landmarks(self, image_bytes: bytes) -> Optional[FacialLandmarks]:
        """Extract facial landmarks from image"""
        analysis = self.analyze_facial_features(image_bytes)
        return analysis.landmarks if analysis else None
    
    def get_face_pose(self, image_bytes: bytes) -> Optional[Dict[str, float]]:
        """Get head pose information"""
        analysis = self.analyze_facial_features(image_bytes)
        return analysis.pose if analysis else None
    
    def get_eye_states(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Get eye state information (open/closed)"""
        analysis = self.analyze_facial_features(image_bytes)
        return analysis.eye_states if analysis else None
    
    def calculate_eye_closure_ratio(self, landmarks: FacialLandmarks) -> float:
        """Calculate eye aspect ratio to determine if eyes are closed"""
        if not landmarks or not landmarks.left_eye or not landmarks.right_eye:
            return 0.5  # Default neutral value
        
        def eye_aspect_ratio(eye_points):
            if len(eye_points) < 6:
                return 0.5
            
            # Calculate vertical distances
            vertical_1 = abs(eye_points[1][1] - eye_points[5][1])
            vertical_2 = abs(eye_points[2][1] - eye_points[4][1])
            
            # Calculate horizontal distance
            horizontal = abs(eye_points[0][0] - eye_points[3][0])
            
            # Calculate aspect ratio
            if horizontal > 0:
                return (vertical_1 + vertical_2) / (2.0 * horizontal)
            return 0.5
        
        # Calculate for both eyes
        left_ear = eye_aspect_ratio(landmarks.left_eye)
        right_ear = eye_aspect_ratio(landmarks.right_eye)
        
        # Return average
        return (left_ear + right_ear) / 2.0
    
    def is_service_available(self) -> bool:
        """Check if AWS Rekognition service is available"""
        return self.is_available
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message"""
        return self.last_error
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            'available': self.is_available,
            'last_error': self.last_error,
            'credentials_configured': bool(self.config.AWS_ACCESS_KEY_ID and self.config.AWS_SECRET_ACCESS_KEY),
            'region': self.config.AWS_REGION
        }