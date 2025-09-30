import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Force override of existing env vars

class Config:
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    REKOGNITION_COLLECTION_ID = os.getenv('REKOGNITION_COLLECTION_ID', 'face_collection')
    
    # Local Database Configuration
    DATABASE_PATH = 'users.db'
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'face_recognition')
    MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'users')
    
    # Camera Configuration
    CAMERA_INDEX = 0
    
    # Face Recognition Configuration
    CONFIDENCE_THRESHOLD = 80.0
    FACE_RECOGNITION_TOLERANCE = float(os.getenv('FACE_RECOGNITION_TOLERANCE', '0.6'))
    
    # System Mode Configuration
    FALLBACK_MODE = os.getenv('FALLBACK_MODE', 'auto')  # 'auto', 'aws_only', 'fallback_only'