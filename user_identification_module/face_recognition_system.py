import cv2
import time
from typing import Optional, Dict, Any
from camera import CameraManager
from rekognition_service import RekognitionService
from database import UserDatabase

class FaceRecognitionSystem:
    def __init__(self):
        self.camera = CameraManager()
        self.rekognition = RekognitionService()
        self.database = UserDatabase()
        self.running = False
    
    def start_system(self):
        """Start the face recognition system"""
        if not self.camera.start_camera():
            print("Failed to start camera")
            return
        
        self.running = True
        print("Face Recognition System Started")
        print("Press 'q' to quit, 'a' to add new user, 's' to scan for user")
        
        try:
            while self.running:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # Display the frame
                self.camera.display_frame(frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.scan_for_user(frame)
                elif key == ord('a'):
                    self.add_new_user(frame)
                
        finally:
            self.stop_system()
    
    def scan_for_user(self, frame) -> Optional[Dict[str, Any]]:
        """Scan for existing user in the frame"""
        print("Scanning for user...")
        
        # Convert frame to bytes
        image_bytes = self.camera.frame_to_bytes(frame)
        
        # Check if there are faces in the image
        faces = self.rekognition.detect_faces(image_bytes)
        if not faces:
            print("No faces detected in the image")
            return None
        
        # Search for known faces
        face_id = self.rekognition.search_faces(image_bytes)
        
        if face_id:
            # User found - fetch from database
            user_data = self.database.get_user_by_face_id(face_id)
            if user_data:
                print(f"User recognized: {user_data['name']}")
                print(f"User data: {user_data['data']}")
                return user_data
            else:
                print("Face found in collection but no user data in database")
        else:
            print("No matching user found - this appears to be a new user")
        
        return None
    
    def add_new_user(self, frame) -> bool:
        """Add a new user to the system"""
        print("Adding new user...")
        
        # Get user information
        name = input("Enter user name: ").strip()
        if not name:
            print("Name cannot be empty")
            return False
        
        # Get additional user data
        print("Enter additional user data (JSON format, or press Enter for empty):")
        data_input = input().strip()
        
        try:
            if data_input:
                import json
                user_data = json.loads(data_input)
            else:
                user_data = {}
        except json.JSONDecodeError:
            print("Invalid JSON format, using empty data")
            user_data = {}
        
        # Convert frame to bytes
        image_bytes = self.camera.frame_to_bytes(frame)
        
        # Check if face is detected
        faces = self.rekognition.detect_faces(image_bytes)
        if not faces:
            print("No face detected in the image. Please ensure your face is visible.")
            return False
        
        # Index the face
        external_image_id = f"user_{name}_{int(time.time())}"
        face_id = self.rekognition.index_face(image_bytes, external_image_id)
        
        if face_id:
            # Add to database
            if self.database.add_user(face_id, name, user_data):
                print(f"User '{name}' added successfully!")
                return True
            else:
                print("Failed to add user to database")
                # Clean up - remove from Rekognition collection
                self.rekognition.delete_face(face_id)
        else:
            print("Failed to index face")
        
        return False
    
    def list_users(self):
        """List all users in the system"""
        users = self.database.list_users()
        if users:
            print("\nRegistered Users:")
            for face_id, name in users:
                print(f"- {name} (ID: {face_id})")
        else:
            print("No users registered")
    
    def stop_system(self):
        """Stop the face recognition system"""
        self.running = False
        self.camera.release_camera()
        print("Face Recognition System Stopped")