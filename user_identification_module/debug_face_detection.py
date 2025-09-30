#!/usr/bin/env python3
"""
Debug script to test face detection capabilities
Tests both local OpenCV detection and AWS Rekognition
"""

import cv2
import os
from camera import CameraManager
from rekognition_service import RekognitionService
from config import Config

class DebugFaceDetection:
    def __init__(self):
        self.camera = CameraManager()
        self.rekognition = None
        
        # Try to initialize Rekognition
        try:
            self.rekognition = RekognitionService()
            print("✅ AWS Rekognition service initialized")
        except Exception as e:
            print(f"❌ AWS Rekognition failed to initialize: {e}")
            print("💡 Make sure your AWS credentials are set in .env file")
        
        # Initialize OpenCV face detector as fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✅ OpenCV face detector initialized")
    
    def detect_faces_opencv(self, frame):
        """Detect faces using OpenCV (local detection)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def detect_faces_aws(self, frame):
        """Detect faces using AWS Rekognition"""
        if not self.rekognition:
            return None
        
        try:
            image_bytes = self.camera.frame_to_bytes(frame)
            faces = self.rekognition.detect_faces(image_bytes)
            return faces
        except Exception as e:
            print(f"AWS face detection error: {e}")
            return None
    
    def run_debug_session(self):
        """Run interactive debug session"""
        print("\n" + "="*60)
        print("🔍 Face Detection Debug Session")
        print("="*60)
        print("Controls:")
        print("- Press 'o' to test OpenCV face detection")
        print("- Press 'a' to test AWS Rekognition face detection")
        print("- Press 'c' to check AWS credentials")
        print("- Press 'q' to quit")
        print("="*60)
        
        if not self.camera.start_camera():
            print("❌ Failed to start camera")
            return
        
        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # Draw instructions on frame
                cv2.putText(frame, "Press 'o' for OpenCV, 'a' for AWS, 'c' for credentials, 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.camera.display_frame(frame, "Face Detection Debug")
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    self.test_opencv_detection(frame)
                elif key == ord('a'):
                    self.test_aws_detection(frame)
                elif key == ord('c'):
                    self.check_aws_credentials()
                    
        finally:
            self.camera.release_camera()
    
    def test_opencv_detection(self, frame):
        """Test OpenCV face detection"""
        print("\n🔍 Testing OpenCV face detection...")
        faces = self.detect_faces_opencv(frame)
        
        if len(faces) > 0:
            print(f"✅ OpenCV detected {len(faces)} face(s)")
            
            # Draw rectangles around faces
            frame_copy = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame_copy, f"Face {len(faces)}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            cv2.imshow("OpenCV Face Detection Result", frame_copy)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyWindow("OpenCV Face Detection Result")
        else:
            print("❌ OpenCV: No faces detected")
            print("💡 Try adjusting lighting or face position")
    
    def test_aws_detection(self, frame):
        """Test AWS Rekognition face detection"""
        print("\n🔍 Testing AWS Rekognition face detection...")
        
        if not self.rekognition:
            print("❌ AWS Rekognition not available")
            return
        
        faces = self.detect_faces_aws(frame)
        
        if faces is None:
            print("❌ AWS Rekognition call failed")
            return
        
        if len(faces) > 0:
            print(f"✅ AWS Rekognition detected {len(faces)} face(s)")
            
            # Draw bounding boxes
            frame_copy = frame.copy()
            height, width = frame.shape[:2]
            
            for i, face in enumerate(faces):
                bbox = face['BoundingBox']
                x = int(bbox['Left'] * width)
                y = int(bbox['Top'] * height)
                w = int(bbox['Width'] * width)
                h = int(bbox['Height'] * height)
                
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"AWS Face {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Print face details
                confidence = face.get('Confidence', 0)
                print(f"  Face {i+1}: Confidence {confidence:.1f}%")
            
            cv2.imshow("AWS Rekognition Result", frame_copy)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyWindow("AWS Rekognition Result")
        else:
            print("❌ AWS Rekognition: No faces detected")
    
    def check_aws_credentials(self):
        """Check AWS credentials setup"""
        print("\n🔧 Checking AWS credentials...")
        
        # Check environment variables
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION')
        
        print(f"AWS_ACCESS_KEY_ID: {'✅ Set' if aws_key and aws_key != 'your_access_key_here' else '❌ Not set'}")
        print(f"AWS_SECRET_ACCESS_KEY: {'✅ Set' if aws_secret and aws_secret != 'your_secret_key_here' else '❌ Not set'}")
        print(f"AWS_REGION: {'✅ Set' if aws_region else '❌ Not set'} ({aws_region})")
        
        if not aws_key or aws_key == 'your_access_key_here':
            print("\n💡 To set up AWS credentials:")
            print("1. Edit the .env file in this directory")
            print("2. Replace 'your_access_key_here' with your actual AWS Access Key ID")
            print("3. Replace 'your_secret_key_here' with your actual AWS Secret Access Key")
            print("4. Make sure your AWS user has Rekognition permissions")

if __name__ == "__main__":
    debug = DebugFaceDetection()
    debug.run_debug_session()