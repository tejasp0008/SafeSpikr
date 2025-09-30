#!/usr/bin/env python3
"""
Visual Demo for Sleep Detection Overlays
Demonstrates all visual overlay features with simulated data
"""

import cv2
import numpy as np
import time
import sys
import os
from datetime import datetime
import math

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_overlay_manager import VisualOverlayManager
from sleep_detection_engine import SleepMetrics, DetectionResult
from aws_sleep_service import FacialLandmarks

class VisualDemo:
    """Visual demonstration of overlay features"""
    
    def __init__(self):
        self.overlay_manager = VisualOverlayManager()
        self.demo_frame_size = (640, 480)
        
    def create_demo_frame(self) -> np.ndarray:
        """Create a demo frame with simulated face"""
        frame = np.zeros((self.demo_frame_size[1], self.demo_frame_size[0], 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(frame.shape[0]):
            intensity = int(50 + (y / frame.shape[0]) * 100)
            frame[y, :] = [intensity // 3, intensity // 2, intensity]
        
        # Draw simulated face
        face_center = (320, 240)
        face_width, face_height = 200, 250
        
        # Face outline (oval)
        cv2.ellipse(frame, face_center, (face_width//2, face_height//2), 0, 0, 360, (180, 150, 120), -1)
        cv2.ellipse(frame, face_center, (face_width//2, face_height//2), 0, 0, 360, (200, 170, 140), 3)
        
        # Eyes
        left_eye_center = (face_center[0] - 40, face_center[1] - 30)
        right_eye_center = (face_center[0] + 40, face_center[1] - 30)
        
        cv2.ellipse(frame, left_eye_center, (25, 15), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, right_eye_center, (25, 15), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, left_eye_center, 8, (50, 50, 50), -1)
        cv2.circle(frame, right_eye_center, 8, (50, 50, 50), -1)
        
        # Nose
        nose_points = np.array([
            [face_center[0], face_center[1] - 10],
            [face_center[0] - 8, face_center[1] + 20],
            [face_center[0] + 8, face_center[1] + 20]
        ], np.int32)
        cv2.fillPoly(frame, [nose_points], (160, 130, 100))
        
        # Mouth
        mouth_center = (face_center[0], face_center[1] + 50)
        cv2.ellipse(frame, mouth_center, (30, 15), 0, 0, 180, (120, 80, 80), -1)
        
        return frame
    
    def create_demo_landmarks(self) -> FacialLandmarks:
        """Create demo facial landmarks"""
        landmarks_data = {
            'leftEye': [
                {'X': 0.44, 'Y': 0.42}, {'X': 0.46, 'Y': 0.40}, {'X': 0.48, 'Y': 0.40},
                {'X': 0.50, 'Y': 0.42}, {'X': 0.48, 'Y': 0.44}, {'X': 0.46, 'Y': 0.44}
            ],
            'rightEye': [
                {'X': 0.56, 'Y': 0.42}, {'X': 0.58, 'Y': 0.40}, {'X': 0.60, 'Y': 0.40},
                {'X': 0.62, 'Y': 0.42}, {'X': 0.60, 'Y': 0.44}, {'X': 0.58, 'Y': 0.44}
            ],
            'nose': [
                {'X': 0.53, 'Y': 0.48}, {'X': 0.52, 'Y': 0.52}, {'X': 0.54, 'Y': 0.52}
            ],
            'mouth': [
                {'X': 0.50, 'Y': 0.60}, {'X': 0.52, 'Y': 0.58}, {'X': 0.54, 'Y': 0.60},
                {'X': 0.52, 'Y': 0.62}
            ]
        }
        
        return FacialLandmarks(landmarks_data)
    
    def run_state_demo(self):
        """Demonstrate different detection states"""
        print("🎭 State Detection Demo")
        print("Cycling through different detection states...")
        
        states = [
            ('normal', 85.0, "Normal alert state"),
            ('drowsy', 72.0, "Drowsiness detected - frequent blinking"),
            ('sleeping', 95.0, "Sleep detected - eyes closed"),
            ('distracted', 78.0, "Distraction detected - head turned away")
        ]
        
        for state, confidence, description in states:
            print(f"\n📊 Demonstrating: {description}")
            
            # Create demo metrics based on state
            if state == 'normal':
                metrics = SleepMetrics(
                    eye_closure_duration=0.2,
                    blink_rate=12.0,
                    head_movement_angle=3.0,
                    drowsiness_score=15.0,
                    distraction_score=10.0,
                    attention_score=90.0,
                    eye_aspect_ratio=0.32,
                    head_stability=0.95
                )
            elif state == 'drowsy':
                metrics = SleepMetrics(
                    eye_closure_duration=1.5,
                    blink_rate=25.0,
                    head_movement_angle=8.0,
                    drowsiness_score=65.0,
                    distraction_score=20.0,
                    attention_score=60.0,
                    eye_aspect_ratio=0.25,
                    head_stability=0.80
                )
            elif state == 'sleeping':
                metrics = SleepMetrics(
                    eye_closure_duration=4.2,
                    blink_rate=3.0,
                    head_movement_angle=2.0,
                    drowsiness_score=85.0,
                    distraction_score=5.0,
                    attention_score=20.0,
                    eye_aspect_ratio=0.12,
                    head_stability=0.90
                )
            else:  # distracted
                metrics = SleepMetrics(
                    eye_closure_duration=0.3,
                    blink_rate=15.0,
                    head_movement_angle=22.0,
                    drowsiness_score=25.0,
                    distraction_score=75.0,
                    attention_score=45.0,
                    eye_aspect_ratio=0.30,
                    head_stability=0.40
                )
            
            # Create detection result
            result = DetectionResult(
                state=state,
                confidence=confidence,
                metrics=metrics,
                timestamp=datetime.now()
            )
            
            # Create demo frame and landmarks
            demo_frame = self.create_demo_frame()
            landmarks = self.create_demo_landmarks()
            
            # System info
            system_info = {
                'detection_method': 'demo',
                'processing_fps': 30.0,
                'detection_accuracy': 92.5,
                'aws_available': True,
                'opencv_available': True,
                'camera_available': True
            }
            
            # Create overlay
            overlay_frame = self.overlay_manager.create_comprehensive_overlay(
                demo_frame, result, landmarks, system_info
            )
            
            # Display for 3 seconds
            cv2.imshow('Sleep Detection Visual Demo', overlay_frame)
            
            # Wait for 3 seconds or key press
            key = cv2.waitKey(3000) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause until any key
    
    def run_metrics_demo(self):
        """Demonstrate metrics visualization"""
        print("\n📊 Metrics Visualization Demo")
        print("Showing animated metrics changes...")
        
        demo_frame = self.create_demo_frame()
        landmarks = self.create_demo_landmarks()
        
        system_info = {
            'detection_method': 'demo',
            'processing_fps': 30.0,
            'detection_accuracy': 92.5,
            'aws_available': True,
            'opencv_available': True,
            'camera_available': True
        }
        
        # Animate metrics over time
        for i in range(100):
            # Create animated metrics
            t = i / 10.0  # Time parameter
            
            metrics = SleepMetrics(
                eye_closure_duration=max(0, 2.0 + math.sin(t * 0.5) * 1.5),
                blink_rate=15.0 + math.sin(t * 0.8) * 10.0,
                head_movement_angle=abs(math.sin(t * 0.3) * 20.0),
                drowsiness_score=50.0 + math.sin(t * 0.4) * 30.0,
                distraction_score=30.0 + math.sin(t * 0.6) * 25.0,
                attention_score=70.0 + math.sin(t * 0.2) * 20.0,
                eye_aspect_ratio=0.25 + math.sin(t * 0.7) * 0.1,
                head_stability=0.7 + math.sin(t * 0.1) * 0.2
            )
            
            # Determine state based on metrics
            if metrics.eye_closure_duration > 3.0:
                state, confidence = 'sleeping', 90.0
            elif metrics.drowsiness_score > 60.0:
                state, confidence = 'drowsy', 75.0
            elif metrics.distraction_score > 50.0:
                state, confidence = 'distracted', 80.0
            else:
                state, confidence = 'normal', 85.0
            
            result = DetectionResult(
                state=state,
                confidence=confidence,
                metrics=metrics,
                timestamp=datetime.now()
            )
            
            # Create overlay
            overlay_frame = self.overlay_manager.create_comprehensive_overlay(
                demo_frame, result, landmarks, system_info
            )
            
            cv2.imshow('Sleep Detection Metrics Demo', overlay_frame)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause
    
    def run_overlay_options_demo(self):
        """Demonstrate different overlay options"""
        print("\n🎨 Overlay Options Demo")
        print("Cycling through different overlay configurations...")
        
        demo_frame = self.create_demo_frame()
        landmarks = self.create_demo_landmarks()
        
        metrics = SleepMetrics(
            eye_closure_duration=1.2,
            blink_rate=18.0,
            head_movement_angle=12.0,
            drowsiness_score=45.0,
            distraction_score=30.0,
            attention_score=75.0,
            eye_aspect_ratio=0.28,
            head_stability=0.85
        )
        
        result = DetectionResult(
            state='drowsy',
            confidence=72.0,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        system_info = {
            'detection_method': 'demo',
            'processing_fps': 30.0,
            'detection_accuracy': 92.5,
            'aws_available': True,
            'opencv_available': True,
            'camera_available': True
        }
        
        # Different overlay configurations
        configs = [
            ("Full Overlay", {'landmarks': True, 'metrics': True, 'alerts': True, 'fps': True, 'timestamp': True}),
            ("Minimal Overlay", {'landmarks': False, 'metrics': False, 'alerts': True, 'fps': False, 'timestamp': False}),
            ("Metrics Only", {'landmarks': False, 'metrics': True, 'alerts': False, 'fps': True, 'timestamp': False}),
            ("Landmarks Only", {'landmarks': True, 'metrics': False, 'alerts': False, 'fps': False, 'timestamp': True}),
        ]
        
        for config_name, options in configs:
            print(f"📋 Showing: {config_name}")
            
            # Set overlay options
            self.overlay_manager.set_overlay_options(**options)
            
            # Create overlay
            overlay_frame = self.overlay_manager.create_comprehensive_overlay(
                demo_frame, result, landmarks, system_info
            )
            
            # Add configuration name
            cv2.putText(overlay_frame, f"Config: {config_name}", (10, overlay_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Sleep Detection Overlay Options', overlay_frame)
            
            key = cv2.waitKey(3000) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        
        # Reset to full overlay
        self.overlay_manager.set_overlay_options(landmarks=True, metrics=True, alerts=True, fps=True, timestamp=True)
    
    def run_interactive_demo(self):
        """Run interactive demo with keyboard controls"""
        print("\n🎮 Interactive Demo")
        print("Controls:")
        print("  1-4: Change detection state")
        print("  L: Toggle landmarks")
        print("  M: Toggle metrics")
        print("  A: Toggle alerts")
        print("  Space: Pause/Resume")
        print("  Q: Quit")
        
        demo_frame = self.create_demo_frame()
        landmarks = self.create_demo_landmarks()
        
        current_state = 'normal'
        paused = False
        
        system_info = {
            'detection_method': 'interactive',
            'processing_fps': 30.0,
            'detection_accuracy': 92.5,
            'aws_available': True,
            'opencv_available': True,
            'camera_available': True
        }
        
        while True:
            if not paused:
                # Create metrics based on current state
                if current_state == 'normal':
                    metrics = SleepMetrics(eye_closure_duration=0.2, blink_rate=12.0, head_movement_angle=3.0,
                                         drowsiness_score=15.0, distraction_score=10.0, attention_score=90.0,
                                         eye_aspect_ratio=0.32, head_stability=0.95)
                    confidence = 85.0
                elif current_state == 'drowsy':
                    metrics = SleepMetrics(eye_closure_duration=1.5, blink_rate=25.0, head_movement_angle=8.0,
                                         drowsiness_score=65.0, distraction_score=20.0, attention_score=60.0,
                                         eye_aspect_ratio=0.25, head_stability=0.80)
                    confidence = 72.0
                elif current_state == 'sleeping':
                    metrics = SleepMetrics(eye_closure_duration=4.2, blink_rate=3.0, head_movement_angle=2.0,
                                         drowsiness_score=85.0, distraction_score=5.0, attention_score=20.0,
                                         eye_aspect_ratio=0.12, head_stability=0.90)
                    confidence = 95.0
                else:  # distracted
                    metrics = SleepMetrics(eye_closure_duration=0.3, blink_rate=15.0, head_movement_angle=22.0,
                                         drowsiness_score=25.0, distraction_score=75.0, attention_score=45.0,
                                         eye_aspect_ratio=0.30, head_stability=0.40)
                    confidence = 78.0
                
                result = DetectionResult(state=current_state, confidence=confidence, metrics=metrics, timestamp=datetime.now())
                
                # Create overlay
                overlay_frame = self.overlay_manager.create_comprehensive_overlay(
                    demo_frame, result, landmarks, system_info
                )
                
                # Add instructions
                instructions = [
                    "1-4: States  L: Landmarks  M: Metrics  A: Alerts  Space: Pause  Q: Quit",
                    f"Current State: {current_state.upper()}"
                ]
                
                for i, instruction in enumerate(instructions):
                    cv2.putText(overlay_frame, instruction, (10, overlay_frame.shape[0] - 50 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Sleep Detection Interactive Demo', overlay_frame)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_state = 'normal'
                print(f"State changed to: {current_state}")
            elif key == ord('2'):
                current_state = 'drowsy'
                print(f"State changed to: {current_state}")
            elif key == ord('3'):
                current_state = 'sleeping'
                print(f"State changed to: {current_state}")
            elif key == ord('4'):
                current_state = 'distracted'
                print(f"State changed to: {current_state}")
            elif key == ord('l'):
                self.overlay_manager.toggle_landmarks()
                print(f"Landmarks: {'ON' if self.overlay_manager.show_landmarks else 'OFF'}")
            elif key == ord('m'):
                self.overlay_manager.toggle_metrics()
                print(f"Metrics: {'ON' if self.overlay_manager.show_metrics else 'OFF'}")
            elif key == ord('a'):
                self.overlay_manager.toggle_alerts()
                print(f"Alerts: {'ON' if self.overlay_manager.show_alerts else 'OFF'}")
            elif key == ord(' '):
                paused = not paused
                print(f"Demo {'PAUSED' if paused else 'RESUMED'}")
    
    def run_all_demos(self):
        """Run all demo scenarios"""
        print("🎬 Sleep Detection Visual Overlay Demo")
        print("=" * 50)
        
        try:
            self.run_state_demo()
            self.run_metrics_demo()
            self.run_overlay_options_demo()
            self.run_interactive_demo()
            
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        finally:
            cv2.destroyAllWindows()
            print("\n👋 Visual demo completed")

def main():
    """Main demo function"""
    demo = VisualDemo()
    
    print("Sleep Detection Visual Demo")
    print("=" * 30)
    print("1. State Detection Demo")
    print("2. Metrics Animation Demo")
    print("3. Overlay Options Demo")
    print("4. Interactive Demo")
    print("5. Run All Demos")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-5): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                demo.run_state_demo()
            elif choice == '2':
                demo.run_metrics_demo()
            elif choice == '3':
                demo.run_overlay_options_demo()
            elif choice == '4':
                demo.run_interactive_demo()
            elif choice == '5':
                demo.run_all_demos()
                break
            else:
                print("Invalid choice. Please select 0-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting demo")
            break
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()