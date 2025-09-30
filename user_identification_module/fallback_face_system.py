from typing import Optional, Dict, Any, List
import time
from opencv_face_service import OpenCVFaceService
from mongodb_service import MongoDBService
from database import UserDatabase
from rekognition_service import RekognitionService
from config import Config

class FallbackFaceSystem:
    """
    Unified face recognition system that automatically falls back to OpenCV + MongoDB
    when AWS Rekognition is not available
    """
    
    def __init__(self):
        self.mode = None
        self.aws_service = None
        self.opencv_service = None
        self.mongodb_service = None
        self.sqlite_service = None
        
        # Initialize services based on configuration
        self._initialize_services()
        self._determine_mode()
        
        print(f"🚀 Face Recognition System initialized in {self.mode.upper()} mode")
    
    def _initialize_services(self):
        """Initialize all available services"""
        # Always initialize OpenCV service (local fallback)
        self.opencv_service = OpenCVFaceService()
        
        # Initialize MongoDB service
        self.mongodb_service = MongoDBService()
        
        # Initialize SQLite service (always available)
        self.sqlite_service = UserDatabase()
        
        # Try to initialize AWS service
        try:
            self.aws_service = RekognitionService()
            print("✅ AWS Rekognition service available")
        except Exception as e:
            print(f"❌ AWS Rekognition not available: {e}")
            self.aws_service = None
    
    def _determine_mode(self):
        """Determine which mode to operate in"""
        fallback_mode = Config.FALLBACK_MODE.lower()
        
        if fallback_mode == 'aws_only':
            if self.aws_service and self._test_aws_service():
                self.mode = 'aws'
            else:
                print("⚠️ AWS_ONLY mode requested but AWS not working, switching to fallback")
                self.mode = 'fallback'
        elif fallback_mode == 'fallback_only':
            self.mode = 'fallback'
        else:  # auto mode
            if self.aws_service and self._test_aws_service():
                self.mode = 'aws'
            else:
                self.mode = 'fallback'
    
    def _test_aws_service(self):
        """Test if AWS service is actually working"""
        if not self.aws_service:
            return False
        
        try:
            # Try to describe the collection (lightweight test)
            self.aws_service.client.describe_collection(CollectionId=self.aws_service.collection_id)
            return True
        except Exception as e:
            print(f"⚠️ AWS service test failed: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components"""
        return {
            'mode': self.mode,
            'aws_available': self.aws_service is not None,
            'mongodb_available': self.mongodb_service.is_connected() if self.mongodb_service else False,
            'opencv_available': self.opencv_service is not None,
            'sqlite_available': True  # Always available
        }
    
    def detect_faces(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Detect faces using the active service"""
        if self.mode == 'aws' and self.aws_service:
            try:
                faces = self.aws_service.detect_faces(image_bytes)
                return faces
            except Exception as e:
                print(f"AWS detection failed, permanently switching to fallback mode: {e}")
                self.mode = 'fallback'  # Permanently switch to fallback
                return self.opencv_service.detect_faces(image_bytes)
        else:
            return self.opencv_service.detect_faces(image_bytes)
    
    def search_faces(self, image_bytes: bytes) -> Optional[str]:
        """Search for known faces"""
        if self.mode == 'aws' and self.aws_service:
            try:
                return self.aws_service.search_faces(image_bytes)
            except Exception as e:
                print(f"AWS search failed, permanently switching to fallback mode: {e}")
                self.mode = 'fallback'  # Permanently switch to fallback
                return self._search_faces_opencv(image_bytes)
        else:
            return self._search_faces_opencv(image_bytes)
    
    def _search_faces_opencv(self, image_bytes: bytes) -> Optional[str]:
        """Search faces using OpenCV and MongoDB/SQLite"""
        # Extract face encoding from image
        face_encoding = self.opencv_service.extract_face_encoding(image_bytes)
        if not face_encoding:
            return None
        
        # Get known users from database
        if self.mongodb_service.is_connected():
            known_users = self.mongodb_service.get_all_users()
        else:
            # Try SQLite with face encodings stored in user data
            try:
                sqlite_users = self.sqlite_service.list_users()
                known_users = []
                
                for face_id, name in sqlite_users:
                    user_data = self.sqlite_service.get_user_by_face_id(face_id)
                    if user_data and 'face_encoding' in user_data['data']:
                        known_users.append({
                            'face_id': face_id,
                            'name': name,
                            'face_encoding': user_data['data']['face_encoding']
                        })
                
                if not known_users:
                    print("⚠️ No users with face encodings found in SQLite")
                    return None
                    
            except Exception as e:
                print(f"⚠️ Error retrieving users from SQLite: {e}")
                return None
        
        # Search for matching face
        return self.opencv_service.search_faces_by_encoding(face_encoding, known_users)
    
    def index_face(self, image_bytes: bytes, external_image_id: str) -> Optional[str]:
        """Add a face to the recognition system"""
        if self.mode == 'aws' and self.aws_service:
            try:
                return self.aws_service.index_face(image_bytes, external_image_id)
            except Exception as e:
                print(f"AWS indexing failed, permanently switching to fallback mode: {e}")
                self.mode = 'fallback'  # Permanently switch to fallback
                return self._index_face_opencv(image_bytes, external_image_id)
        else:
            return self._index_face_opencv(image_bytes, external_image_id)
    
    def _index_face_opencv(self, image_bytes: bytes, external_image_id: str) -> Optional[str]:
        """Index face using OpenCV"""
        face_id, face_encoding = self.opencv_service.index_face(image_bytes, external_image_id)
        return face_id if face_encoding else None
    
    def add_user(self, face_id: str, name: str, user_data: Dict[Any, Any], face_encoding: List[float] = None) -> bool:
        """Add user to the database"""
        if self.mode == 'aws':
            # Use SQLite for AWS mode
            return self.sqlite_service.add_user(face_id, name, user_data)
        else:
            # Use MongoDB for fallback mode (includes face encoding)
            if self.mongodb_service.is_connected() and face_encoding:
                return self.mongodb_service.add_user(face_id, name, face_encoding, user_data)
            else:
                # Fallback to SQLite
                return self.sqlite_service.add_user(face_id, name, user_data)
    
    def get_user_by_face_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by face ID"""
        if self.mode == 'aws':
            return self.sqlite_service.get_user_by_face_id(face_id)
        else:
            if self.mongodb_service.is_connected():
                return self.mongodb_service.get_user_by_face_id(face_id)
            else:
                return self.sqlite_service.get_user_by_face_id(face_id)
    
    def list_users(self) -> List[tuple]:
        """List all users"""
        if self.mode == 'aws':
            return self.sqlite_service.list_users()
        else:
            if self.mongodb_service.is_connected():
                return self.mongodb_service.list_users()
            else:
                return self.sqlite_service.list_users()
    
    def add_user_complete(self, image_bytes: bytes, name: str, user_data: Dict[Any, Any]) -> Dict[str, Any]:
        """Complete user addition process"""
        # Detect faces first
        faces = self.detect_faces(image_bytes)
        if not faces:
            return {
                'success': False,
                'message': 'No face detected in the image. Please ensure your face is visible.'
            }
        
        # Generate external image ID
        external_image_id = f"user_{name}_{int(time.time())}"
        
        if self.mode == 'aws' and self.aws_service:
            # AWS mode
            try:
                face_id = self.aws_service.index_face(image_bytes, external_image_id)
                if face_id:
                    if self.sqlite_service.add_user(face_id, name, user_data):
                        return {
                            'success': True,
                            'message': f"User '{name}' added successfully using AWS Rekognition!",
                            'face_id': face_id,
                            'mode': 'aws'
                        }
                    else:
                        # Clean up
                        self.aws_service.delete_face(face_id)
                        return {'success': False, 'message': 'Failed to add user to database'}
                else:
                    return {'success': False, 'message': 'Failed to index face with AWS Rekognition'}
            except Exception as e:
                print(f"AWS failed, falling back to OpenCV: {e}")
                return self._add_user_opencv(image_bytes, name, user_data, external_image_id)
        else:
            # Fallback mode
            return self._add_user_opencv(image_bytes, name, user_data, external_image_id)
    
    def _add_user_opencv(self, image_bytes: bytes, name: str, user_data: Dict[Any, Any], external_image_id: str) -> Dict[str, Any]:
        """Add user using OpenCV + MongoDB/SQLite"""
        # Extract face encoding
        face_encoding = self.opencv_service.extract_face_encoding(image_bytes)
        if not face_encoding:
            return {'success': False, 'message': 'Failed to extract face features'}
        
        # Generate face ID
        face_id = self.opencv_service.generate_face_id()
        
        # Add to database
        if self.mongodb_service.is_connected():
            if self.mongodb_service.add_user(face_id, name, face_encoding, user_data):
                return {
                    'success': True,
                    'message': f"User '{name}' added successfully using OpenCV + MongoDB!",
                    'face_id': face_id,
                    'mode': 'fallback_mongodb'
                }
            else:
                return {'success': False, 'message': 'Failed to add user to MongoDB'}
        else:
            # Fallback to SQLite (store face encoding in user data for basic recognition)
            user_data_with_encoding = user_data.copy()
            user_data_with_encoding['face_encoding'] = face_encoding
            
            if self.sqlite_service.add_user(face_id, name, user_data_with_encoding):
                return {
                    'success': True,
                    'message': f"User '{name}' added using OpenCV + SQLite (basic recognition available)",
                    'face_id': face_id,
                    'mode': 'fallback_sqlite'
                }
            else:
                return {'success': False, 'message': 'Failed to add user to database'}
    
    def scan_for_user(self, image_bytes: bytes) -> Dict[str, Any]:
        """Scan for existing user"""
        # Detect faces first
        faces = self.detect_faces(image_bytes)
        if not faces:
            return {'success': False, 'message': 'No faces detected in the image'}
        
        # Search for known faces
        face_id = self.search_faces(image_bytes)
        
        if face_id:
            # User found - fetch from database
            user_data = self.get_user_by_face_id(face_id)
            if user_data:
                return {
                    'success': True,
                    'user_found': True,
                    'user': user_data,
                    'message': f"User recognized: {user_data['name']} (Mode: {self.mode})",
                    'mode': self.mode
                }
            else:
                return {
                    'success': False,
                    'message': 'Face found in collection but no user data in database'
                }
        else:
            return {
                'success': True,
                'user_found': False,
                'message': f'Face detected ({len(faces)} face(s)) - this appears to be a new user (Mode: {self.mode})',
                'mode': self.mode
            }