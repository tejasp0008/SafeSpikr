#!/usr/bin/env python3
"""
Simple test to verify the face recognition system is working
"""

import cv2
import time
from camera import CameraManager
from fallback_face_system import FallbackFaceSystem

def test_system():
    print("🚀 Testing Face Recognition System")
    print("=" * 50)
    
    # Initialize system
    system = FallbackFaceSystem()
    status = system.get_system_status()
    
    print(f"📊 System Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  OpenCV: {'✅' if status['opencv_available'] else '❌'}")
    print(f"  SQLite: {'✅' if status['sqlite_available'] else '❌'}")
    
    # Initialize camera
    camera = CameraManager()
    if not camera.start_camera():
        print("❌ Failed to start camera")
        return False
    
    print("\n📹 Camera started successfully")
    print("Position your face in front of the camera and press Enter...")
    input()
    
    # Capture frame
    frame = camera.capture_frame()
    if frame is None:
        print("❌ Failed to capture frame")
        camera.release_camera()
        return False
    
    # Convert to bytes
    image_bytes = camera.frame_to_bytes(frame)
    
    # Test face detection
    print("🔍 Testing face detection...")
    try:
        result = system.scan_for_user(image_bytes)
        print(f"📊 Scan result: {result}")
        
        if result['success']:
            if result.get('user_found'):
                print(f"✅ User recognized: {result['user']['name']}")
            else:
                print("✅ Face detected but user not found (new user)")
                
                # Test adding a user
                print("\n➕ Testing user addition...")
                test_result = system.add_user_complete(
                    image_bytes, 
                    "Test User", 
                    {"test": True, "timestamp": time.time()}
                )
                print(f"📊 Add user result: {test_result}")
                
                if test_result['success']:
                    print("✅ User added successfully!")
                    
                    # Test recognition again
                    print("\n🔍 Testing recognition of newly added user...")
                    recognition_result = system.scan_for_user(image_bytes)
                    print(f"📊 Recognition result: {recognition_result}")
                    
                    if recognition_result.get('user_found'):
                        print("✅ User successfully recognized after addition!")
                    else:
                        print("⚠️ User not recognized after addition (may need better face matching)")
        else:
            print(f"❌ Face detection failed: {result['message']}")
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        camera.release_camera()
    
    print("\n🏁 Test completed")

if __name__ == "__main__":
    test_system()