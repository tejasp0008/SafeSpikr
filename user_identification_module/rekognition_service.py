import boto3
from botocore.exceptions import ClientError
from typing import Optional, List, Dict, Any
from config import Config

class RekognitionService:
    def __init__(self):
        self.client = boto3.client(
            'rekognition',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION
        )
        self.collection_id = Config.REKOGNITION_COLLECTION_ID
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.describe_collection(CollectionId=self.collection_id)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                try:
                    self.client.create_collection(CollectionId=self.collection_id)
                    print(f"Created collection: {self.collection_id}")
                except ClientError as create_error:
                    print(f"Error creating collection: {create_error}")
            else:
                print(f"Error checking collection: {e}")
    
    def detect_faces(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Detect faces in the image"""
        try:
            response = self.client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            return response.get('FaceDetails', [])
        except ClientError as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def search_faces(self, image_bytes: bytes) -> Optional[str]:
        """Search for known faces in the collection"""
        try:
            response = self.client.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                FaceMatchThreshold=self.confidence_threshold,
                MaxFaces=1
            )
            
            matches = response.get('FaceMatches', [])
            if matches:
                return matches[0]['Face']['FaceId']
            return None
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidParameterException':
                # No face detected in image
                return None
            print(f"Error searching faces: {e}")
            return None
    
    def index_face(self, image_bytes: bytes, external_image_id: str) -> Optional[str]:
        """Add a face to the collection"""
        try:
            response = self.client.index_faces(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                ExternalImageId=external_image_id,
                MaxFaces=1,
                QualityFilter='AUTO'
            )
            
            face_records = response.get('FaceRecords', [])
            if face_records:
                return face_records[0]['Face']['FaceId']
            return None
            
        except ClientError as e:
            print(f"Error indexing face: {e}")
            return None
    
    def delete_face(self, face_id: str) -> bool:
        """Delete a face from the collection"""
        try:
            self.client.delete_faces(
                CollectionId=self.collection_id,
                FaceIds=[face_id]
            )
            return True
        except ClientError as e:
            print(f"Error deleting face: {e}")
            return False