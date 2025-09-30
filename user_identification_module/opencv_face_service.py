import cv2
import face_recognition
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class OpenCVFaceService:
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.tolerance = Config.FACE_RECOGNITION_TOLERANCE
        print("✅ OpenCV Face Service initialized")
    
    def detect_faces(self, image) -> List[Dict[str, Any]]:
        """Detect faces in image using OpenCV"""
        if isinstance(image, bytes):
            # Convert bytes to numpy array
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert to format similar to AWS Rekognition
        face_details = []
        height, width = image.shape[:2]
        
        for (x, y, w, h) in faces:
            face_detail = {
                'BoundingBox': {
                    'Left': x / width,
                    'Top': y / height,
                    'Width': w / width,
                    'Height': h / height
                },
                'Confidence': 95.0  # OpenCV doesn't provide confidence, so we use a default
            }
            face_details.append(face_detail)
        
        return face_details
    
    def extract_face_encoding(self, image) -> Optional[List[float]]:
        """Extract face encoding using face_recognition library"""
        try:
            if isinstance(image, bytes):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB (face_recognition expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return None
            
            # Get encoding for the first face found
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if face_encodings:
                return face_encodings[0].tolist()  # Convert numpy array to list
            
            return None
            
        except Exception as e:
            print(f"Error extracting face encoding: {e}")
            return None
    
    def compare_faces(self, known_encodings: List[List[float]], face_encoding: List[float]) -> Tuple[List[bool], List[float]]:
        """Compare face encoding with known encodings"""
        try:
            if not known_encodings or not face_encoding:
                return [], []
            
            # Convert to numpy arrays
            known_encodings_np = np.array(known_encodings)
            face_encoding_np = np.array(face_encoding).reshape(1, -1)
            
            # Calculate cosine similarity (face_recognition uses euclidean distance, but cosine works well too)
            similarities = cosine_similarity(face_encoding_np, known_encodings_np)[0]
            
            # Convert similarities to distances (1 - similarity)
            distances = 1 - similarities
            
            # Check if distances are within tolerance
            matches = distances <= self.tolerance
            
            return matches.tolist(), distances.tolist()
            
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return [], []
    
    def find_best_match(self, known_encodings: List[List[float]], face_encoding: List[float]) -> Tuple[Optional[int], float]:
        """Find the best matching face encoding"""
        matches, distances = self.compare_faces(known_encodings, face_encoding)
        
        if not matches or not any(matches):
            return None, float('inf')
        
        # Find the best match (lowest distance)
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]
        
        if matches[best_match_index]:
            return best_match_index, best_distance
        
        return None, float('inf')
    
    def generate_face_id(self) -> str:
        """Generate a unique face ID"""
        return f"opencv_{uuid.uuid4().hex[:16]}"
    
    def search_faces_by_encoding(self, face_encoding: List[float], known_users: List[Dict[str, Any]]) -> Optional[str]:
        """Search for a face in known users by encoding"""
        if not known_users or not face_encoding:
            return None
        
        known_encodings = []
        face_ids = []
        
        for user in known_users:
            if 'face_encoding' in user and user['face_encoding']:
                known_encodings.append(user['face_encoding'])
                face_ids.append(user['face_id'])
        
        if not known_encodings:
            return None
        
        best_match_index, distance = self.find_best_match(known_encodings, face_encoding)
        
        if best_match_index is not None:
            return face_ids[best_match_index]
        
        return None
    
    def index_face(self, image, external_image_id: str) -> Tuple[Optional[str], Optional[List[float]]]:
        """Index a face (extract encoding and generate face ID)"""
        face_encoding = self.extract_face_encoding(image)
        
        if face_encoding:
            face_id = self.generate_face_id()
            return face_id, face_encoding
        
        return None, None