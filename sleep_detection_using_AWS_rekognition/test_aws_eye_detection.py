#!/usr/bin/env python3
"""
Test AWS Eye Detection
Test AWS Rekognition's ability to detect closed eyes
"""

import cv2
import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_manager import CameraManager
from aws_sleep_service import AWSRekognitionSleepService

def main():
    """Test AWS eye detection"""
    print("👁️ AWS Eye Detection Test")
    print("=" * 30)
    
    camera_manager = CameraManager()
    aws_service = AWSRekognitionSleepService()
    
    # Check AWS availability
    if not aws_service.is_service_available():
        print(f"❌ AWS Rekognition not available: {aws_service.get_last_error()}")
        return
    
    print("✅ AWS Rekognition is available")
    
    if not camera_manager.start_camera():
        print("❌ Could not start camera")
        return
    
    print("🎬 Test started. Close and open your eyes to test detection.")
    print("Press 'q' to quit.")
    
    try:
        while True:
            frame = camera_manager.capture_frame()
            
            if frame is not None:
                # Convert frame to bytes for AWS
                image_bytes = camera_manager.frame_to_bytes(frame)
                
                if image_bytes:
                    # Analyze with AWS
                    analysis = aws_service.analyze_facial_features(image_bytes)
                    
                    display_frame = frame.copy()
                    
                    if analysis:
                        # Extract eye state information
                        eyes_open = True
                        eye_confidence = 0.0
                        
                        if analysis.eye_states and 'eyes_open' in analysis.eye_states:
                            eye_state_data = analysis.eye_states['eyes_open']
                            if isinstance(eye_state_data, dict):
                                eyes_open = eye_state_data.get('Value', True)
                                eye_confidence = eye_state_data.get('Confidence', 0.0)
                            else:
                                eyes_open = bool(eye_state_data)
                        
                        # Calculate eye aspect ratio
                        ear = 0.3
                        if analysis.landmarks:
                            ear = aws_service.calculate_eye_closure_ratio(analysis.landmarks)
                        
                        # Determine eye state
                        if not eyes_open and eye_confidence > 60:
                            eye_state = "CLOSED"
                            color = (0, 0, 255)  # Red
                        elif eyes_open and eye_confidence > 60:
                            eye_state = "OPEN"
                            color = (0, 255, 0)  # Green
                        else:
                            eye_state = "UNCERTAIN"
                            color = (0, 255, 255)  # Yellow
                        
                        # Display information
                        cv2.putText(display_frame, f"AWS Eyes: {eye_state}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(display_frame, f"Confidence: {eye_confidence:.1f}%", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"EAR: {ear:.3f}", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Face Confidence: {analysis.confidence:.1f}%", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        # Console output
                        print(f"\rEye State: {eye_state:10} | Confidence: {eye_confidence:5.1f}% | EAR: {ear:.3f}", end="")
                        
                        # Draw facial landmarks if available
                        if analysis.landmarks:
                            height, width = frame.shape[:2]
                            
                            # Draw eye landmarks
                            for eye_points in [analysis.landmarks.left_eye, analysis.landmarks.right_eye]:
                                if eye_points:
                                    for x, y in eye_points:
                                        px = int(x * width)
                                        py = int(y * height)
                                        cv2.circle(display_frame, (px, py), 2, (255, 255, 0), -1)
                    else:
                        cv2.putText(display_frame, "No face detected by AWS", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print("\rNo face detected by AWS", end="")
                else:
                    cv2.putText(display_frame, "Frame conversion failed", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('AWS Eye Detection Test', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    finally:
        camera_manager.release_camera()
        cv2.destroyAllWindows()
        print("\n✅ AWS eye detection test completed")

if __name__ == '__main__':
    main()