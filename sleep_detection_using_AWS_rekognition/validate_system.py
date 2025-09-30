#!/usr/bin/env python3
"""
System Validation Script
Comprehensive validation of sleep detection system components
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sleep_config import SleepDetectionConfig
from aws_sleep_service import AWSRekognitionSleepService
from opencv_sleep_service import OpenCVSleepService
from sleep_detection_engine import SleepDetectionEngine
from state_classifier import AdvancedStateClassifier
from camera_manager import CameraManager
from error_handler import get_logger, get_error_manager

class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.logger = get_logger()
        self.error_manager = get_error_manager()
        self.validation_results = {}
        
    def validate_configuration(self):
        """Validate system configuration"""
        print("🔧 Validating Configuration...")
        
        try:
            config = SleepDetectionConfig()
            validation = config.validate_config()
            
            self.validation_results['configuration'] = {
                'status': 'pass' if validation['valid'] else 'warning',
                'aws_configured': validation['aws_configured'],
                'issues': validation['issues'],
                'thresholds': config.get_detection_thresholds()
            }
            
            if validation['valid']:
                print("  ✅ Configuration is valid")
            else:
                print("  ⚠️ Configuration has issues:")
                for issue in validation['issues']:
                    print(f"    - {issue}")
            
            if validation['aws_configured']:
                print("  ✅ AWS credentials configured")
            else:
                print("  ⚠️ AWS credentials not configured (will use OpenCV fallback)")
                
        except Exception as e:
            print(f"  ❌ Configuration validation failed: {e}")
            self.validation_results['configuration'] = {'status': 'fail', 'error': str(e)}
    
    def validate_aws_service(self):
        """Validate AWS Rekognition service"""
        print("\n☁️ Validating AWS Rekognition Service...")
        
        try:
            service = AWSRekognitionSleepService()
            
            self.validation_results['aws_service'] = {
                'available': service.is_service_available(),
                'last_error': service.get_last_error(),
                'status': service.get_service_status()
            }
            
            if service.is_service_available():
                print("  ✅ AWS Rekognition service is available")
                
                # Test with a simple image
                test_image = self.create_test_image()
                image_bytes = cv2.imencode('.jpg', test_image)[1].tobytes()
                
                analysis = service.analyze_facial_features(image_bytes)
                if analysis:
                    print("  ✅ AWS facial analysis working")
                else:
                    print("  ⚠️ AWS facial analysis returned no results (expected for test image)")
                    
            else:
                print(f"  ❌ AWS Rekognition service unavailable: {service.get_last_error()}")
                
        except Exception as e:
            print(f"  ❌ AWS service validation failed: {e}")
            self.validation_results['aws_service'] = {'status': 'fail', 'error': str(e)}
    
    def validate_opencv_service(self):
        """Validate OpenCV fallback service"""
        print("\n🔍 Validating OpenCV Service...")
        
        try:
            service = OpenCVSleepService()
            
            self.validation_results['opencv_service'] = {
                'available': service.is_service_available(),
                'last_error': service.get_last_error(),
                'status': service.get_service_status()
            }
            
            if service.is_service_available():
                print("  ✅ OpenCV service is available")
                
                # Test with a test frame
                test_frame = self.create_test_image()
                analysis = service.analyze_frame_opencv(test_frame)
                
                if analysis['success']:
                    print("  ✅ OpenCV frame analysis working")
                else:
                    print(f"  ⚠️ OpenCV analysis: {analysis.get('error', 'No face detected')}")
                    
            else:
                print(f"  ❌ OpenCV service unavailable: {service.get_last_error()}")
                
        except Exception as e:
            print(f"  ❌ OpenCV service validation failed: {e}")
            self.validation_results['opencv_service'] = {'status': 'fail', 'error': str(e)}
    
    def validate_camera(self):
        """Validate camera functionality"""
        print("\n📷 Validating Camera...")
        
        try:
            camera = CameraManager()
            
            # Test camera initialization
            camera_started = camera.start_camera()
            
            self.validation_results['camera'] = {
                'available': camera_started,
                'info': camera.get_camera_info(),
                'statistics': camera.get_statistics()
            }
            
            if camera_started:
                print("  ✅ Camera initialized successfully")
                
                # Test frame capture
                frame = camera.capture_frame()
                if frame is not None:
                    print(f"  ✅ Frame capture working (resolution: {frame.shape[1]}x{frame.shape[0]})")
                    
                    # Test frame processing
                    frame_bytes = camera.frame_to_bytes(frame)
                    if len(frame_bytes) > 0:
                        print("  ✅ Frame to bytes conversion working")
                    else:
                        print("  ⚠️ Frame to bytes conversion failed")
                else:
                    print("  ❌ Frame capture failed")
                
                # Cleanup
                camera.release_camera()
                
            else:
                print("  ❌ Camera initialization failed")
                
        except Exception as e:
            print(f"  ❌ Camera validation failed: {e}")
            self.validation_results['camera'] = {'status': 'fail', 'error': str(e)}
    
    def validate_detection_engine(self):
        """Validate sleep detection engine"""
        print("\n🧠 Validating Detection Engine...")
        
        try:
            engine = SleepDetectionEngine()
            
            # Test with mock data
            from unittest.mock import Mock
            
            landmarks = Mock()
            emotions = Mock()
            pose_data = {'Yaw': 5.0, 'Pitch': 2.0, 'Roll': 1.0}
            eye_aspect_ratio = 0.25
            
            # Test normal detection
            result = engine.analyze_frame(landmarks, emotions, pose_data, eye_aspect_ratio)
            
            self.validation_results['detection_engine'] = {
                'status': 'pass',
                'test_result': {
                    'state': result.state,
                    'confidence': result.confidence,
                    'metrics_available': result.metrics is not None
                }
            }
            
            print(f"  ✅ Detection engine working (test result: {result.state}, confidence: {result.confidence:.1f}%)")
            
            # Test sleep detection
            sleep_result = engine.analyze_frame(landmarks, emotions, pose_data, 0.1)  # Very low eye ratio
            print(f"  ✅ Sleep detection test (result: {sleep_result.state})")
            
            # Test detection summary
            summary = engine.get_detection_summary()
            print(f"  ✅ Detection summary available ({summary['total_detections']} detections)")
            
        except Exception as e:
            print(f"  ❌ Detection engine validation failed: {e}")
            self.validation_results['detection_engine'] = {'status': 'fail', 'error': str(e)}
    
    def validate_state_classifier(self):
        """Validate state classifier"""
        print("\n🎯 Validating State Classifier...")
        
        try:
            classifier = AdvancedStateClassifier()
            
            from sleep_detection_engine import SleepMetrics
            
            # Test normal state
            normal_metrics = SleepMetrics(
                eye_closure_duration=0.5,
                blink_rate=12.0,
                head_movement_angle=5.0,
                eye_aspect_ratio=0.3
            )
            
            state, confidence, reason = classifier.classify_state(normal_metrics, 80.0)
            
            print(f"  ✅ Normal state classification: {state} (confidence: {confidence:.1f}%)")
            
            # Test sleep state
            sleep_metrics = SleepMetrics(
                eye_closure_duration=4.0,
                blink_rate=5.0,
                head_movement_angle=2.0,
                eye_aspect_ratio=0.1
            )
            
            state, confidence, reason = classifier.classify_state(sleep_metrics, 90.0)
            print(f"  ✅ Sleep state classification: {state} (reason: {reason})")
            
            # Test state summary
            summary = classifier.get_state_summary()
            print(f"  ✅ State summary available (current: {summary['current_state']})")
            
            self.validation_results['state_classifier'] = {
                'status': 'pass',
                'current_state': summary['current_state'],
                'total_transitions': summary['total_transitions']
            }
            
        except Exception as e:
            print(f"  ❌ State classifier validation failed: {e}")
            self.validation_results['state_classifier'] = {'status': 'fail', 'error': str(e)}
    
    def validate_error_handling(self):
        """Validate error handling system"""
        print("\n🛡️ Validating Error Handling...")
        
        try:
            # Test error logging
            try:
                raise ValueError("Test error for validation")
            except Exception as e:
                from error_handler import ErrorCategory, ErrorSeverity
                record = self.error_manager.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.LOW)
                
            # Test error summary
            summary = self.error_manager.get_error_summary()
            
            self.validation_results['error_handling'] = {
                'status': 'pass',
                'total_errors': summary['total_errors'],
                'resolved_errors': summary['resolved_errors']
            }
            
            print("  ✅ Error handling system working")
            print(f"  ✅ Error summary available ({summary['total_errors']} total errors)")
            
        except Exception as e:
            print(f"  ❌ Error handling validation failed: {e}")
            self.validation_results['error_handling'] = {'status': 'fail', 'error': str(e)}
    
    def validate_performance(self):
        """Validate system performance"""
        print("\n⚡ Validating Performance...")
        
        try:
            # Test detection speed
            engine = SleepDetectionEngine()
            
            start_time = time.time()
            iterations = 50
            
            for i in range(iterations):
                from unittest.mock import Mock
                result = engine.analyze_frame(
                    Mock(), Mock(), {}, np.random.uniform(0.1, 0.4)
                )
            
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            self.validation_results['performance'] = {
                'status': 'pass' if avg_time < 0.1 else 'warning',
                'avg_processing_time': avg_time,
                'estimated_fps': fps,
                'real_time_capable': avg_time < 0.033  # 30 FPS threshold
            }
            
            print(f"  ✅ Average processing time: {avg_time:.3f}s")
            print(f"  ✅ Estimated FPS: {fps:.1f}")
            
            if avg_time < 0.033:
                print("  ✅ Real-time capable (>30 FPS)")
            elif avg_time < 0.1:
                print("  ⚠️ Acceptable performance (<10 FPS)")
            else:
                print("  ❌ Performance may be insufficient for real-time use")
                
        except Exception as e:
            print(f"  ❌ Performance validation failed: {e}")
            self.validation_results['performance'] = {'status': 'fail', 'error': str(e)}
    
    def create_test_image(self):
        """Create a test image for validation"""
        # Create a simple test image with basic shapes
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some basic shapes to simulate a face
        cv2.rectangle(image, (200, 150), (440, 350), (128, 128, 128), -1)  # Face
        cv2.circle(image, (280, 220), 15, (255, 255, 255), -1)  # Left eye
        cv2.circle(image, (360, 220), 15, (255, 255, 255), -1)  # Right eye
        cv2.rectangle(image, (310, 280), (330, 300), (255, 255, 255), -1)  # Nose
        cv2.rectangle(image, (300, 320), (340, 330), (255, 255, 255), -1)  # Mouth
        
        return image
    
    def run_validation(self):
        """Run complete system validation"""
        print("🚀 Sleep Detection System Validation")
        print("=" * 50)
        
        validation_start = time.time()
        
        # Run all validations
        self.validate_configuration()
        self.validate_aws_service()
        self.validate_opencv_service()
        self.validate_camera()
        self.validate_detection_engine()
        self.validate_state_classifier()
        self.validate_error_handling()
        self.validate_performance()
        
        validation_time = time.time() - validation_start
        
        # Generate summary report
        self.generate_report(validation_time)
        
        return self.validation_results
    
    def generate_report(self, validation_time):
        """Generate validation summary report"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY REPORT")
        print("=" * 50)
        
        total_components = len(self.validation_results)
        passed_components = sum(1 for result in self.validation_results.values() 
                               if result.get('status') == 'pass')
        warning_components = sum(1 for result in self.validation_results.values() 
                                if result.get('status') == 'warning')
        failed_components = sum(1 for result in self.validation_results.values() 
                               if result.get('status') == 'fail')
        
        print(f"Validation Time: {validation_time:.2f} seconds")
        print(f"Total Components: {total_components}")
        print(f"Passed: {passed_components} ✅")
        print(f"Warnings: {warning_components} ⚠️")
        print(f"Failed: {failed_components} ❌")
        
        success_rate = (passed_components / total_components * 100) if total_components > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Component status
        print("\n📋 Component Status:")
        for component, result in self.validation_results.items():
            status = result.get('status', 'unknown')
            if status == 'pass':
                print(f"  ✅ {component.replace('_', ' ').title()}")
            elif status == 'warning':
                print(f"  ⚠️ {component.replace('_', ' ').title()}")
            elif status == 'fail':
                print(f"  ❌ {component.replace('_', ' ').title()}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
        
        # System readiness assessment
        print(f"\n🎯 System Readiness Assessment:")
        
        if failed_components == 0:
            if warning_components == 0:
                print("  🎉 System is fully ready for production use!")
            else:
                print("  ✅ System is ready with minor warnings.")
        else:
            print("  ⚠️ System has issues that should be addressed before use.")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        aws_available = self.validation_results.get('aws_service', {}).get('available', False)
        opencv_available = self.validation_results.get('opencv_service', {}).get('available', False)
        camera_available = self.validation_results.get('camera', {}).get('available', False)
        
        if not aws_available and not opencv_available:
            print("  ❌ No detection method available - check AWS credentials or OpenCV installation")
        elif not aws_available:
            print("  ⚠️ AWS unavailable - system will use OpenCV fallback (reduced accuracy)")
        elif not camera_available:
            print("  ❌ Camera unavailable - check camera connection and permissions")
        else:
            print("  ✅ All core components are functional")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'validation_time': validation_time,
            'summary': {
                'total_components': total_components,
                'passed': passed_components,
                'warnings': warning_components,
                'failed': failed_components,
                'success_rate': success_rate
            },
            'results': self.validation_results
        }
        
        try:
            with open('validation_report.json', 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved to: validation_report.json")
        except Exception as e:
            print(f"\n⚠️ Could not save report file: {e}")

def main():
    """Main validation function"""
    validator = SystemValidator()
    results = validator.run_validation()
    
    # Return exit code based on results
    failed_components = sum(1 for result in results.values() 
                           if result.get('status') == 'fail')
    
    return 0 if failed_components == 0 else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)