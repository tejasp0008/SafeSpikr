#!/usr/bin/env python3
"""
AWS-only Face Recognition System
Simplified version that only uses AWS Rekognition + SQLite
"""

import time
from typing import Optional, Dict, Any
from rekognition_service import RekognitionService
from database import UserDatabase
from camera import CameraManager

class AWSFaceSystem:
    """AWS-only face recognition system"""
    
    def __init__(self):
        print("🚀 Initializing AWS-only Face Recognition System...")
        
        # Initialize services
        self.rekognition = RekognitionService()
        self.database = UserDatabase()
        self.camera = CameraManager()
        
        print("✅ AWS Face Recognition System ready!")
    
    def detect_faces(self, image_bytes: bytes):
        """Detect faces using AWS Rekognition"""
        return self.rekognition.detect_faces(image_bytes)
    
    def search_faces(self, image_bytes: bytes) -> Optional[str]:
        """Search for known faces using AWS Rekognition"""
        return self.rekognition.search_faces(image_bytes)
    
    def index_face(self, image_bytes: bytes, external_image_id: str) -> Optional[str]:
        """Add face to AWS Rekognition collection"""
        return self.rekognition.index_face(image_bytes, external_image_id)
    
    def add_user(self, face_id: str, name: str, user_data: Dict[Any, Any]) -> bool:
        """Add user to SQLite database"""
        return self.database.add_user(face_id, name, user_data)
    
    def get_user_by_face_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by face ID"""
        return self.database.get_user_by_face_id(face_id)
    
    def list_users(self):
        """List all users"""
        return self.database.list_users()
    
    def scan_for_user(self, image_bytes: bytes) -> Dict[str, Any]:
        """Scan for existing user using AWS"""
        # Detect faces first
        faces = self.detect_faces(image_bytes)
        if not faces:
            return {'success': False, 'message': 'No faces detected in the image'}
        
        print(f"🔍 AWS detected {len(faces)} face(s)")
        
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
                    'message': f"User recognized: {user_data['name']} (AWS Rekognition)",
                    'mode': 'aws'
                }
            else:
                return {
                    'success': False,
                    'message': 'Face found in AWS collection but no user data in database'
                }
        else:
            return {
                'success': True,
                'user_found': False,
                'message': f'Face detected ({len(faces)} face(s)) - this appears to be a new user (AWS Rekognition)',
                'mode': 'aws'
            }
    
    def add_user_complete(self, image_bytes: bytes, name: str, user_data: Dict[Any, Any]) -> Dict[str, Any]:
        """Complete user addition process using AWS"""
        # Detect faces first
        faces = self.detect_faces(image_bytes)
        if not faces:
            return {
                'success': False,
                'message': 'No face detected in the image. Please ensure your face is visible.'
            }
        
        print(f"🔍 AWS detected {len(faces)} face(s) for user addition")
        
        # Generate external image ID
        external_image_id = f"user_{name}_{int(time.time())}"
        
        # Index the face in AWS
        face_id = self.index_face(image_bytes, external_image_id)
        
        if face_id:
            print(f"✅ Face indexed in AWS with ID: {face_id}")
            
            # Add to database
            if self.add_user(face_id, name, user_data):
                return {
                    'success': True,
                    'message': f"User '{name}' added successfully using AWS Rekognition!",
                    'face_id': face_id,
                    'mode': 'aws'
                }
            else:
                # Clean up - remove from Rekognition collection
                self.rekognition.delete_face(face_id)
                return {'success': False, 'message': 'Failed to add user to database'}
        else:
            return {'success': False, 'message': 'Failed to index face with AWS Rekognition'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'mode': 'aws',
            'aws_available': True,
            'mongodb_available': False,
            'opencv_available': False,
            'sqlite_available': True
        }