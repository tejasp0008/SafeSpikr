#!/usr/bin/env python3
"""
Test Closed Eye Detection
Specifically test the system's ability to detect closed eyes
"""

import cv2
import numpy as np
import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_manager import CameraManager
from opencv_sleep_service import OpenCVSleepService
from sleep_detection_engine import SleepDetectionEngine, SleepMetrics
from state_classifier import AdvancedStateClassifier

class ClosedEyeDetectionTest:
    """Test closed eye detection capabilities"""
    
    def __init__(self):
        self.camera_manager = CameraManager()
        self.opencv_service = OpenCVSleepService()
        self.detection_engine = SleepDetectionEngine()
        self.state_classifier = AdvancedStateClassifier()
        
        # Detection history
        self.detection_history = []
        
    def run_closed_eye_test(self):
        """Run interactive closed eye detection test"""
        print("👁️ Closed Eye Detection Test")
        print("=" * 40)
        print("This test will help verify closed eye detection.")
        print("\nInstructions:")
        print("1. Position yourself in front of the camera")
        print("2. Keep your eyes OPEN for 10 seconds")
        print("3. CLOSE your eyes for 10 seconds")
        print("4. OPEN your eyes again")
        print("5. Press 'q' to quit anytime")
        
        input("\nPress Enter when ready...")
        
        if not self.camera_manager.start_camera():
            print("❌ Could not start camera")
            return
        
        print("\n🎬 Test started! Follow the instructions above.")
        
        try:
            test_phase = "OPEN"  # Start with eyes open
            phase_start_time = time.time()
            phase_duration = 10  # 10 seconds per phase
            
            while True:
                frame = self.camera_manager.capture_frame()
                
                if frame is not None:
                    # Test detection
                    detection_result = self._test_frame_detection(frame)
                    
                    # Create display frame
                    display_frame = frame.copy()
                    
                    # Calculate remaining time for current phase
                    elapsed = time.time() - phase_start_time
                    remaining = max(0, phase_duration - elapsed)
                    
                    # Switch phases
                    if remaining <= 0:
                        if test_phase == "OPEN":
                            test_phase = "CLOSED"
                            print(f"\n👁️ Now CLOSE your eyes for {phase_duration} seconds!")
                        elif test_phase == "CLOSED":
                            test_phase = "OPEN"
                            print(f"\n👁️ Now OPEN your eyes for {phase_duration} seconds!")
                        
                        phase_start_time = time.time()
                        remaining = phase_duration
                    
                    # Display current phase and countdown
                    phase_color = (0, 255, 0) if test_phase == "OPEN" else (0, 0, 255)
                    cv2.putText(display_frame, f"Phase: {test_phase} EYES", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, phase_color, 2)
                    cv2.putText(display_frame, f"Time: {remaining:.1f}s", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Display detection results
                    if detection_result:
                        state = detection_result.get('state', 'unknown')
                        confidence = detection_result.get('confidence', 0)
                        ear = detection_result.get('eye_aspect_ratio', 0)
                        face_detected = detection_result.get('face_detected', False)
                        
                        # Color based on detection accuracy
                        if test_phase == "OPEN" and state in ['normal', 'drowsy']:
                            result_color = (0, 255, 0)  # Green - correct
                        elif test_phase == "CLOSED" and state == 'sleeping':
                            result_color = (0, 255, 0)  # Green - correct
                        else:
                            result_color = (0, 255, 255)  # Yellow - incorrect/uncertain
                        
                        cv2.putText(display_frame, f"Detected: {state.upper()}", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                        cv2.putText(display_frame, f"Confidence: {confidence:.1f}%", (10, 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(display_frame, f"EAR: {ear:.3f}", (10, 170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(display_frame, f"Face: {'YES' if face_detected else 'NO'}", (10, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                   (0, 255, 0) if face_detected else (0, 0, 255), 1)
                        
                        # Store detection for analysis
                        self.detection_history.append({
                            'timestamp': time.time(),
                            'expected_phase': test_phase,
                            'detected_state': state,
                            'confidence': confidence,
                            'ear': ear,
                            'face_detected': face_detected
                        })
                    else:
                        cv2.putText(display_frame, "No Detection", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show frame
                    cv2.imshow('Closed Eye Detection Test', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nTest interrupted")
        finally:
            self.camera_manager.release_camera()
            cv2.destroyAllWindows()
            
            # Analyze results
            self._analyze_test_results()
    
    def _test_frame_detection(self, frame):
        """Test detection on a single frame"""
        try:
            # Test OpenCV detection
            landmarks = self.opencv_service.detect_face_landmarks(frame)
            
            if landmarks and landmarks.left_eye and landmarks.right_eye:
                # Calculate eye aspect ratio
                left_ear = self.opencv_service.calculate_eye_aspect_ratio(landmarks.left_eye)
                right_ear = self.opencv_service.calculate_eye_aspect_ratio(landmarks.right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Create metrics
                metrics = SleepMetrics(
                    eye_closure_duration=2.0 if avg_ear < 0.2 else 0.1,
                    blink_rate=10.0,
                    head_movement_angle=5.0,
                    eye_aspect_ratio=avg_ear
                )
                
                # Run through detection engine
                result = self.detection_engine.analyze_frame(
                    landmarks=landmarks,
                    emotions=None,
                    pose_data={'Yaw': 0, 'Pitch': 0, 'Roll': 0},
                    eye_aspect_ratio=avg_ear
                )
                
                # Apply state classification
                state, confidence, reason = self.state_classifier.classify_state(
                    result.metrics, result.confidence
                )
                
                return {
                    'state': state,
                    'confidence': confidence,
                    'eye_aspect_ratio': avg_ear,
                    'face_detected': True,
                    'reason': reason
                }
            else:
                return {
                    'state': 'no_face',
                    'confidence': 0,
                    'eye_aspect_ratio': 0,
                    'face_detected': False,
                    'reason': 'No face detected'
                }
                
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def _analyze_test_results(self):
        """Analyze test results and provide feedback"""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS ANALYSIS")
        print("=" * 50)
        
        if not self.detection_history:
            print("❌ No detection data collected")
            return
        
        # Separate results by expected phase
        open_eye_results = [r for r in self.detection_history if r['expected_phase'] == 'OPEN']
        closed_eye_results = [r for r in self.detection_history if r['expected_phase'] == 'CLOSED']
        
        print(f"Total detections: {len(self.detection_history)}")
        print(f"Open eye phase: {len(open_eye_results)} detections")
        print(f"Closed eye phase: {len(closed_eye_results)} detections")
        
        # Analyze open eye detection accuracy
        if open_eye_results:
            open_correct = sum(1 for r in open_eye_results 
                             if r['detected_state'] in ['normal', 'drowsy'] and r['face_detected'])
            open_accuracy = (open_correct / len(open_eye_results)) * 100
            avg_open_ear = sum(r['ear'] for r in open_eye_results) / len(open_eye_results)
            
            print(f"\n👁️ OPEN EYES DETECTION:")
            print(f"  Accuracy: {open_accuracy:.1f}% ({open_correct}/{len(open_eye_results)})")
            print(f"  Average EAR: {avg_open_ear:.3f}")
            print(f"  Face detection rate: {sum(1 for r in open_eye_results if r['face_detected']) / len(open_eye_results) * 100:.1f}%")
        
        # Analyze closed eye detection accuracy
        if closed_eye_results:
            closed_correct = sum(1 for r in closed_eye_results 
                               if r['detected_state'] == 'sleeping' and r['face_detected'])
            closed_accuracy = (closed_correct / len(closed_eye_results)) * 100
            avg_closed_ear = sum(r['ear'] for r in closed_eye_results) / len(closed_eye_results)
            
            print(f"\n👁️ CLOSED EYES DETECTION:")
            print(f"  Accuracy: {closed_accuracy:.1f}% ({closed_correct}/{len(closed_eye_results)})")
            print(f"  Average EAR: {avg_closed_ear:.3f}")
            print(f"  Face detection rate: {sum(1 for r in closed_eye_results if r['face_detected']) / len(closed_eye_results) * 100:.1f}%")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        face_detection_rate = sum(1 for r in self.detection_history if r['face_detected']) / len(self.detection_history) * 100
        print(f"  Face detection rate: {face_detection_rate:.1f}%")
        
        if face_detection_rate < 50:
            print("  ❌ Poor face detection - check lighting and camera position")
        elif face_detection_rate < 80:
            print("  ⚠️ Moderate face detection - consider improving lighting")
        else:
            print("  ✅ Good face detection rate")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if closed_eye_results:
            avg_closed_ear = sum(r['ear'] for r in closed_eye_results) / len(closed_eye_results)
            if avg_closed_ear > 0.25:
                print("  - Eye aspect ratio threshold may need adjustment")
                print(f"  - Consider lowering EAR threshold below {avg_closed_ear:.3f}")
        
        if face_detection_rate < 80:
            print("  - Improve lighting conditions")
            print("  - Position face more directly toward camera")
            print("  - Ensure camera is at eye level")
        
        print("\n" + "=" * 50)

def main():
    """Main test function"""
    tester = ClosedEyeDetectionTest()
    tester.run_closed_eye_test()

if __name__ == '__main__':
    main()