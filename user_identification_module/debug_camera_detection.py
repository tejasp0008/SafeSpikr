#!/usr/bin/env python3
"""
Debug script specifically for camera and face detection issues
"""

import cv2
import numpy as np
from camera import CameraManager
from fallback_face_system import FallbackFaceSystem
import time

def test_camera_basic():
    """Test basic camera functionality"""
    print("🔍 Testing basic camera functionality...")
    
    camera = CameraManager()
    if not camera.start_camera():
        print("❌ Failed to start camera")
        return False
    
    print("✅ Camera started successfully")
    
    # Capture a few frames
    for i in range(5):
        frame = camera.capture_frame()
        if frame is not None:
            print(f"✅ Frame {i+1}: {frame.shape} - OK")
        else:
            print(f"❌ Frame {i+1}: Failed to capture")
            camera.release_camera()
            return False
        time.sleep(0.5)
    
    camera.release_camera()
    print("✅ Basic camera test passed")
    return True

def test_opencv_face_detection():
    """Test OpenCV face detection with live camera"""
    print("\n🔍 Testing OpenCV face detection...")
    
    camera = CameraManager()
    if not camera.start_camera():
        print("❌ Failed to start camera")
        return False
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("📹 Starting live face detection test...")
    print("Position your face in front of the camera")
    print("Press 'q' to quit, 's' to save detection result")
    
    detection_count = 0
    frame_count = 0
    
    try:
        while True:
            frame = camera.capture_frame()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Face {len(faces)}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                detection_count += 1
            
            # Show status
            status_text = f"Frames: {frame_count}, Detections: {detection_count}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("OpenCV Face Detection Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) > 0:
                # Save frame with detection
                cv2.imwrite("debug_face_detection.jpg", frame)
                print(f"💾 Saved frame with {len(faces)} face(s) detected")
                
    finally:
        camera.release_camera()
        cv2.destroyAllWindows()
    
    print(f"📊 Detection Summary: {detection_count} detections in {frame_count} frames")
    return detection_count > 0

def test_aws_face_detection():
    """Test AWS face detection"""
    print("\n🔍 Testing AWS face detection...")
    
    camera = CameraManager()
    if not camera.start_camera():
        print("❌ Failed to start camera")
        return False
    
    system = FallbackFaceSystem()
    
    print("📹 Capture a frame for AWS testing...")
    print("Position your face and press Enter...")
    input()
    
    frame = camera.capture_frame()
    if frame is None:
        print("❌ Failed to capture frame")
        camera.release_camera()
        return False
    
    # Convert to bytes
    image_bytes = camera.frame_to_bytes(frame)
    
    # Test AWS detection
    try:
        faces = system.detect_faces(image_bytes)
        print(f"✅ AWS detected {len(faces)} face(s)")
        
        if faces:
            for i, face in enumerate(faces):
                confidence = face.get('Confidence', 0)
                bbox = face.get('BoundingBox', {})
                print(f"  Face {i+1}: Confidence {confidence:.1f}%")
                print(f"    BoundingBox: {bbox}")
        
        camera.release_camera()
        return len(faces) > 0
        
    except Exception as e:
        print(f"❌ AWS detection error: {e}")
        camera.release_camera()
        return False

def test_face_recognition_library():
    """Test face_recognition library"""
    print("\n🔍 Testing face_recognition library...")
    
    camera = CameraManager()
    if not camera.start_camera():
        print("❌ Failed to start camera")
        return False
    
    import face_recognition
    
    print("📹 Capture a frame for face_recognition testing...")
    print("Position your face and press Enter...")
    input()
    
    frame = camera.capture_frame()
    if frame is None:
        print("❌ Failed to capture frame")
        camera.release_camera()
        return False
    
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"✅ face_recognition found {len(face_locations)} face(s)")
        
        if face_locations:
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print(f"✅ Generated {len(face_encodings)} face encoding(s)")
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                print(f"  Face {i+1}: Location ({top}, {right}, {bottom}, {left})")
        
        camera.release_camera()
        return len(face_locations) > 0
        
    except Exception as e:
        print(f"❌ face_recognition error: {e}")
        camera.release_camera()
        return False

def test_system_integration():
    """Test the full system integration"""
    print("\n🔍 Testing full system integration...")
    
    system = FallbackFaceSystem()
    status = system.get_system_status()
    
    print(f"📊 System Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  AWS Available: {status['aws_available']}")
    print(f"  MongoDB Available: {status['mongodb_available']}")
    print(f"  OpenCV Available: {status['opencv_available']}")
    print(f"  SQLite Available: {status['sqlite_available']}")
    
    camera = CameraManager()
    if not camera.start_camera():
        print("❌ Failed to start camera")
        return False
    
    print("\n📹 Testing system scan_for_user...")
    print("Position your face and press Enter...")
    input()
    
    frame = camera.capture_frame()
    if frame is None:
        print("❌ Failed to capture frame")
        camera.release_camera()
        return False
    
    image_bytes = camera.frame_to_bytes(frame)
    
    try:
        result = system.scan_for_user(image_bytes)
        print(f"📊 Scan Result: {result}")
        
        camera.release_camera()
        return result['success']
        
    except Exception as e:
        print(f"❌ System scan error: {e}")
        import traceback
        traceback.print_exc()
        camera.release_camera()
        return False

def main():
    """Run all tests"""
    print("🚀 Face Detection Debug Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Camera", test_camera_basic),
        ("OpenCV Face Detection", test_opencv_face_detection),
        ("AWS Face Detection", test_aws_face_detection),
        ("face_recognition Library", test_face_recognition_library),
        ("System Integration", test_system_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print("\n💡 Recommendations:")
    if not results.get("Basic Camera", False):
        print("- Check camera permissions and ensure no other app is using the camera")
    if not results.get("OpenCV Face Detection", False):
        print("- Try better lighting conditions")
        print("- Face the camera directly")
        print("- Move closer to the camera")
    if not results.get("AWS Face Detection", False):
        print("- Check AWS credentials in .env file")
        print("- Verify AWS Rekognition permissions")
    if not results.get("face_recognition Library", False):
        print("- face_recognition library may need better lighting")
    if not results.get("System Integration", False):
        print("- Check the full error trace above")

if __name__ == "__main__":
    main()