#!/usr/bin/env python3
"""
Simple Camera Test
Quick test to verify camera is working
"""

import cv2
import sys

def test_camera():
    """Simple camera test"""
    print("📷 Testing Camera Access...")
    
    # Try different camera indices
    for i in range(3):
        print(f"\nTrying camera index {i}...")
        
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✅ Camera {i} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Frame captured: {width}x{height}")
                
                # Show the frame
                cv2.imshow(f'Camera {i} Test', frame)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                cap.release()
                print(f"✅ Camera {i} is working!")
                return i
            else:
                print(f"❌ Camera {i} opened but cannot capture frames")
        else:
            print(f"❌ Camera {i} cannot be opened")
        
        cap.release()
    
    print("\n❌ No working camera found!")
    print("💡 Troubleshooting tips:")
    print("   - Check if camera is connected")
    print("   - Close other applications using the camera")
    print("   - Check camera permissions")
    print("   - Try external USB camera if built-in camera fails")
    
    return None

if __name__ == '__main__':
    working_camera = test_camera()
    
    if working_camera is not None:
        print(f"\n🎉 Success! Use camera index {working_camera}")
        print(f"💡 Update CAMERA_INDEX={working_camera} in .env file if needed")
    else:
        print("\n❌ Camera test failed")
        sys.exit(1)