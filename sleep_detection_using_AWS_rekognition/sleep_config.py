import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv(override=True)

class SleepDetectionConfig:
    """Configuration management for sleep detection module"""
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    
    # Camera Configuration
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
    
    # Sleep Detection Thresholds
    SLEEP_EYE_CLOSURE_THRESHOLD = float(os.getenv('SLEEP_EYE_CLOSURE_THRESHOLD', '3.0'))  # seconds
    DROWSY_BLINK_RATE_THRESHOLD = float(os.getenv('DROWSY_BLINK_RATE_THRESHOLD', '20.0'))  # blinks per minute
    
    # Distraction Detection Thresholds
    DISTRACTION_HEAD_ANGLE_THRESHOLD = float(os.getenv('DISTRACTION_HEAD_ANGLE_THRESHOLD', '15.0'))  # degrees
    DISTRACTION_DURATION_THRESHOLD = float(os.getenv('DISTRACTION_DURATION_THRESHOLD', '5.0'))  # seconds
    
    # Confidence Thresholds
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7'))
    AWS_FACE_CONFIDENCE_THRESHOLD = float(os.getenv('AWS_FACE_CONFIDENCE_THRESHOLD', '80.0'))
    
    # System Configuration
    USE_AWS_PRIMARY = os.getenv('USE_AWS_PRIMARY', 'true').lower() == 'true'
    FALLBACK_TO_OPENCV = os.getenv('FALLBACK_TO_OPENCV', 'true').lower() == 'true'
    FRAME_SKIP_RATE = int(os.getenv('FRAME_SKIP_RATE', '3'))  # Process every 3rd frame
    
    # Web UI Configuration
    WEB_HOST = os.getenv('WEB_HOST', '127.0.0.1')
    WEB_PORT = int(os.getenv('WEB_PORT', '5001'))
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    @classmethod
    def get_aws_credentials(cls) -> Dict[str, str]:
        """Get AWS credentials as dictionary"""
        return {
            'aws_access_key_id': cls.AWS_ACCESS_KEY_ID,
            'aws_secret_access_key': cls.AWS_SECRET_ACCESS_KEY,
            'region_name': cls.AWS_REGION
        }
    
    @classmethod
    def get_detection_thresholds(cls) -> Dict[str, float]:
        """Get all detection thresholds as dictionary"""
        return {
            'sleep_eye_closure_threshold': cls.SLEEP_EYE_CLOSURE_THRESHOLD,
            'drowsy_blink_rate_threshold': cls.DROWSY_BLINK_RATE_THRESHOLD,
            'distraction_head_angle_threshold': cls.DISTRACTION_HEAD_ANGLE_THRESHOLD,
            'distraction_duration_threshold': cls.DISTRACTION_DURATION_THRESHOLD,
            'min_confidence_threshold': cls.MIN_CONFIDENCE_THRESHOLD,
            'aws_face_confidence_threshold': cls.AWS_FACE_CONFIDENCE_THRESHOLD
        }
    
    @classmethod
    def update_threshold(cls, threshold_name: str, value: float) -> bool:
        """Update a specific threshold value"""
        threshold_map = {
            'sleep_eye_closure_threshold': 'SLEEP_EYE_CLOSURE_THRESHOLD',
            'drowsy_blink_rate_threshold': 'DROWSY_BLINK_RATE_THRESHOLD',
            'distraction_head_angle_threshold': 'DISTRACTION_HEAD_ANGLE_THRESHOLD',
            'distraction_duration_threshold': 'DISTRACTION_DURATION_THRESHOLD',
            'min_confidence_threshold': 'MIN_CONFIDENCE_THRESHOLD',
            'aws_face_confidence_threshold': 'AWS_FACE_CONFIDENCE_THRESHOLD'
        }
        
        if threshold_name in threshold_map:
            setattr(cls, threshold_map[threshold_name], value)
            return True
        return False
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        # Check AWS credentials
        if not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY:
            issues.append("AWS credentials not configured")
        
        # Check threshold ranges
        if cls.SLEEP_EYE_CLOSURE_THRESHOLD <= 0:
            issues.append("Sleep eye closure threshold must be positive")
        
        if cls.DROWSY_BLINK_RATE_THRESHOLD <= 0:
            issues.append("Drowsy blink rate threshold must be positive")
        
        if not (0 <= cls.MIN_CONFIDENCE_THRESHOLD <= 1):
            issues.append("Min confidence threshold must be between 0 and 1")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'aws_configured': bool(cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY)
        }