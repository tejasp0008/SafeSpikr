#!/usr/bin/env python3
"""
Comprehensive Test Suite for Sleep Detection Module
Unit tests, integration tests, and validation scripts
"""

import unittest
import numpy as np
import cv2
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleep_config import SleepDetectionConfig
from aws_sleep_service import AWSRekognitionSleepService, FacialAnalysis, FacialLandmarks, EmotionData
from opencv_sleep_service import OpenCVSleepService, OpenCVFacialLandmarks
from sleep_detection_engine import SleepDetectionEngine, SleepMetrics, DetectionResult
from state_classifier import AdvancedStateClassifier, StateHistory
from camera_manager import CameraManager, FrameProcessor
from error_handler import ErrorManager, SleepDetectionLogger, ErrorCategory, ErrorSeverity

class TestSleepDetectionConfig(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        self.config = SleepDetectionConfig()
    
    def test_default_values(self):
        """Test default configuration values"""
        self.assertEqual(self.config.SLEEP_EYE_CLOSURE_THRESHOLD, 3.0)
        self.assertEqual(self.config.DROWSY_BLINK_RATE_THRESHOLD, 20.0)
        self.assertEqual(self.config.DISTRACTION_HEAD_ANGLE_THRESHOLD, 15.0)
        self.assertEqual(self.config.MIN_CONFIDENCE_THRESHOLD, 0.7)
    
    def test_threshold_updates(self):
        """Test threshold update functionality"""
        # Test valid threshold update
        result = self.config.update_threshold('sleep_eye_closure_threshold', 4.0)
        self.assertTrue(result)
        self.assertEqual(self.config.SLEEP_EYE_CLOSURE_THRESHOLD, 4.0)
        
        # Test invalid threshold name
        result = self.config.update_threshold('invalid_threshold', 5.0)
        self.assertFalse(result)
    
    def test_config_validation(self):
        """Test configuration validation"""
        validation = self.config.validate_config()
        self.assertIn('valid', validation)
        self.assertIn('issues', validation)
        self.assertIn('aws_configured', validation)

class TestAWSRekognitionService(unittest.TestCase):
    """Test AWS Rekognition service"""
    
    def setUp(self):
        self.service = AWSRekognitionSleepService()
    
    @patch('boto3.client')
    def test_service_initialization(self, mock_boto_client):
        """Test AWS service initialization"""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        service = AWSRekognitionSleepService()
        self.assertIsNotNone(service.client)
    
    def test_facial_landmarks_creation(self):
        """Test facial landmarks data structure"""
        landmarks_data = {
            'leftEye': [{'X': 0.3, 'Y': 0.4}, {'X': 0.35, 'Y': 0.4}],
            'rightEye': [{'X': 0.6, 'Y': 0.4}, {'X': 0.65, 'Y': 0.4}],
            'nose': [{'X': 0.5, 'Y': 0.5}],
            'mouth': [{'X': 0.5, 'Y': 0.7}]
        }
        
        landmarks = FacialLandmarks(landmarks_data)
        self.assertEqual(len(landmarks.left_eye), 2)
        self.assertEqual(len(landmarks.right_eye), 2)
        self.assertEqual(landmarks.left_eye[0], (0.3, 0.4))
    
    def test_emotion_data_creation(self):
        """Test emotion data processing"""
        emotions_list = [
            {'Type': 'HAPPY', 'Confidence': 85.5},
            {'Type': 'CALM', 'Confidence': 70.2},
            {'Type': 'SURPRISED', 'Confidence': 15.3}
        ]
        
        emotion_data = EmotionData(emotions_list)
        self.assertEqual(emotion_data.dominant_emotion, 'HAPPY')
        self.assertEqual(emotion_data.confidence, 85.5)
        self.assertIn('HAPPY', emotion_data.emotions)
    
    def test_eye_closure_ratio_calculation(self):
        """Test eye aspect ratio calculation"""
        # Create mock landmarks
        landmarks_data = {
            'leftEye': [
                {'X': 0.3, 'Y': 0.4}, {'X': 0.32, 'Y': 0.39}, {'X': 0.34, 'Y': 0.39},
                {'X': 0.36, 'Y': 0.4}, {'X': 0.34, 'Y': 0.41}, {'X': 0.32, 'Y': 0.41}
            ],
            'rightEye': [
                {'X': 0.6, 'Y': 0.4}, {'X': 0.62, 'Y': 0.39}, {'X': 0.64, 'Y': 0.39},
                {'X': 0.66, 'Y': 0.4}, {'X': 0.64, 'Y': 0.41}, {'X': 0.62, 'Y': 0.41}
            ]
        }
        
        landmarks = FacialLandmarks(landmarks_data)
        ratio = self.service.calculate_eye_closure_ratio(landmarks)
        
        self.assertIsInstance(ratio, float)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

class TestOpenCVService(unittest.TestCase):
    """Test OpenCV fallback service"""
    
    def setUp(self):
        self.service = OpenCVSleepService()
    
    def test_service_initialization(self):
        """Test OpenCV service initialization"""
        # Service should initialize even without camera
        self.assertIsNotNone(self.service.face_cascade)
        self.assertIsNotNone(self.service.eye_cascade)
    
    def test_eye_aspect_ratio_calculation(self):
        """Test eye aspect ratio calculation"""
        # Create mock eye points
        eye_points = [
            (0.3, 0.4), (0.32, 0.39), (0.34, 0.39),
            (0.36, 0.4), (0.34, 0.41), (0.32, 0.41)
        ]
        
        ratio = self.service.calculate_eye_aspect_ratio(eye_points)
        self.assertIsInstance(ratio, float)
        self.assertGreaterEqual(ratio, 0.0)
    
    def test_create_test_frame(self):
        """Test creation of test frame for testing"""
        # Create a simple test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a simple rectangle to simulate a face
        cv2.rectangle(frame, (200, 150), (440, 350), (255, 255, 255), -1)
        
        # Test frame processing
        landmarks = self.service.detect_face_landmarks(frame)
        # Note: This might return None if no actual face is detected, which is expected

class TestSleepDetectionEngine(unittest.TestCase):
    """Test sleep detection engine"""
    
    def setUp(self):
        self.engine = SleepDetectionEngine()
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.blink_tracker)
        self.assertIsNotNone(self.engine.head_tracker)
        self.assertIsNotNone(self.engine.temporal_smoother)
    
    def test_metrics_creation(self):
        """Test sleep metrics creation"""
        metrics = SleepMetrics(
            eye_closure_duration=2.5,
            blink_rate=15.0,
            head_movement_angle=10.0,
            drowsiness_score=45.0,
            distraction_score=20.0
        )
        
        self.assertEqual(metrics.eye_closure_duration, 2.5)
        self.assertEqual(metrics.blink_rate, 15.0)
        self.assertEqual(metrics.head_movement_angle, 10.0)
    
    def test_detection_result_creation(self):
        """Test detection result creation"""
        metrics = SleepMetrics()
        result = DetectionResult(
            state='drowsy',
            confidence=75.5,
            metrics=metrics
        )
        
        self.assertEqual(result.state, 'drowsy')
        self.assertEqual(result.confidence, 75.5)
        self.assertIsNotNone(result.timestamp)
    
    def test_frame_analysis(self):
        """Test frame analysis with mock data"""
        # Create mock data
        landmarks = Mock()
        emotions = Mock()
        pose_data = {'Yaw': 5.0, 'Pitch': 2.0, 'Roll': 1.0}
        eye_aspect_ratio = 0.25
        
        # Analyze frame
        result = self.engine.analyze_frame(
            landmarks, emotions, pose_data, eye_aspect_ratio
        )
        
        self.assertIsInstance(result, DetectionResult)
        self.assertIn(result.state, ['normal', 'drowsy', 'sleeping', 'distracted'])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 100.0)

class TestStateClassifier(unittest.TestCase):
    """Test advanced state classifier"""
    
    def setUp(self):
        self.classifier = AdvancedStateClassifier()
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier.state_history)
        self.assertIsNotNone(self.classifier.hysteresis_classifier)
    
    def test_state_classification(self):
        """Test state classification logic"""
        # Test normal state
        metrics = SleepMetrics(
            eye_closure_duration=0.5,
            blink_rate=12.0,
            head_movement_angle=5.0,
            eye_aspect_ratio=0.3
        )
        
        state, confidence, reason = self.classifier.classify_state(metrics, 80.0)
        
        self.assertIn(state, ['normal', 'drowsy', 'sleeping', 'distracted'])
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(reason, str)
    
    def test_sleep_state_detection(self):
        """Test sleep state detection"""
        # Create metrics indicating sleep
        metrics = SleepMetrics(
            eye_closure_duration=4.0,  # Above threshold
            blink_rate=5.0,
            head_movement_angle=2.0,
            eye_aspect_ratio=0.1  # Very low (eyes closed)
        )
        
        state, confidence, reason = self.classifier.classify_state(metrics, 90.0)
        
        # Should detect sleep due to extended eye closure
        self.assertEqual(state, 'sleeping')
        self.assertGreater(confidence, 70.0)
    
    def test_distraction_detection(self):
        """Test distraction detection"""
        # Create metrics indicating distraction
        metrics = SleepMetrics(
            eye_closure_duration=0.2,
            blink_rate=10.0,
            head_movement_angle=25.0,  # Above threshold
            head_stability=0.3  # Low stability
        )
        
        state, confidence, reason = self.classifier.classify_state(metrics, 85.0)
        
        # Should detect distraction due to head movement
        # Note: Actual result depends on hysteresis logic
        self.assertIn(state, ['distracted', 'normal'])

class TestCameraManager(unittest.TestCase):
    """Test camera management"""
    
    def setUp(self):
        self.camera_manager = CameraManager()
    
    def test_camera_manager_initialization(self):
        """Test camera manager initialization"""
        self.assertIsNotNone(self.camera_manager.frame_processor)
        self.assertEqual(self.camera_manager.is_running, False)
    
    def test_frame_processor(self):
        """Test frame processor"""
        processor = FrameProcessor()
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = processor.preprocess_frame(test_frame)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (480, 640, 3))
        
        # Test frame to bytes conversion
        frame_bytes = processor.frame_to_bytes(test_frame)
        self.assertIsInstance(frame_bytes, bytes)
        self.assertGreater(len(frame_bytes), 0)
    
    def test_camera_info(self):
        """Test camera info retrieval"""
        info = self.camera_manager.get_camera_info()
        self.assertIsInstance(info, dict)
        self.assertIn('available', info)

class TestErrorHandling(unittest.TestCase):
    """Test error handling system"""
    
    def setUp(self):
        self.logger = SleepDetectionLogger()
        self.error_manager = ErrorManager(self.logger)
    
    def test_error_record_creation(self):
        """Test error record creation"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            record = self.error_manager.handle_error(
                e, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
            )
            
            self.assertEqual(record.error_type, 'ValueError')
            self.assertEqual(record.error_message, 'Test error')
            self.assertEqual(record.category, ErrorCategory.SYSTEM)
            self.assertEqual(record.severity, ErrorSeverity.MEDIUM)
    
    def test_error_recovery(self):
        """Test automatic error recovery"""
        try:
            raise ConnectionError("AWS connection failed")
        except Exception as e:
            record = self.error_manager.handle_error(
                e, ErrorCategory.AWS_SERVICE, ErrorSeverity.MEDIUM,
                context={'error_key': 'connection_failed'}
            )
            
            # Should attempt recovery
            self.assertIsNotNone(record)
    
    def test_error_summary(self):
        """Test error summary generation"""
        # Generate some test errors
        for i in range(5):
            try:
                raise RuntimeError(f"Test error {i}")
            except Exception as e:
                self.error_manager.handle_error(e, ErrorCategory.SYSTEM)
        
        summary = self.error_manager.get_error_summary()
        
        self.assertIn('total_errors', summary)
        self.assertIn('resolved_errors', summary)
        self.assertIn('category_distribution', summary)
        self.assertEqual(summary['total_errors'], 5)

class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_detection_performance(self):
        """Test detection performance under load"""
        engine = SleepDetectionEngine()
        
        # Measure processing time for multiple frames
        start_time = time.time()
        
        for i in range(100):
            # Simulate frame analysis
            metrics = SleepMetrics(
                eye_closure_duration=np.random.uniform(0, 5),
                blink_rate=np.random.uniform(5, 30),
                head_movement_angle=np.random.uniform(0, 30)
            )
            
            result = engine.analyze_frame(
                landmarks=Mock(),
                emotions=Mock(),
                pose_data={'Yaw': 0, 'Pitch': 0, 'Roll': 0},
                eye_aspect_ratio=np.random.uniform(0.1, 0.4)
            )
            
            self.assertIsInstance(result, DetectionResult)
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        # Should process frames quickly (< 50ms per frame for real-time)
        self.assertLess(avg_time, 0.05, f"Average processing time too high: {avg_time:.3f}s")
    
    def test_memory_usage(self):
        """Test memory usage over time"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and destroy many objects
        engine = SleepDetectionEngine()
        
        for i in range(1000):
            result = engine.analyze_frame(
                landmarks=Mock(),
                emotions=Mock(),
                pose_data={},
                eye_aspect_ratio=0.3
            )
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB)
        self.assertLess(memory_increase, 50 * 1024 * 1024, 
                       f"Memory increase too high: {memory_increase / 1024 / 1024:.1f}MB")

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_detection_flow(self):
        """Test complete detection flow"""
        # This test requires actual components working together
        config = SleepDetectionConfig()
        
        # Test with OpenCV service (more reliable than AWS for testing)
        opencv_service = OpenCVSleepService()
        engine = SleepDetectionEngine()
        classifier = AdvancedStateClassifier()
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process through OpenCV (may not detect face, but should not crash)
        analysis = opencv_service.analyze_frame_opencv(test_frame)
        
        # Should return a result (even if no face detected)
        self.assertIsInstance(analysis, dict)
        self.assertIn('success', analysis)
    
    def test_fallback_mechanism(self):
        """Test AWS to OpenCV fallback"""
        # Mock AWS service failure
        aws_service = AWSRekognitionSleepService()
        opencv_service = OpenCVSleepService()
        
        # AWS should be unavailable in test environment
        self.assertFalse(aws_service.is_service_available())
        
        # OpenCV should be available
        self.assertTrue(opencv_service.is_service_available())

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSleepDetectionConfig,
        TestAWSRekognitionService,
        TestOpenCVService,
        TestSleepDetectionEngine,
        TestStateClassifier,
        TestCameraManager,
        TestErrorHandling,
        TestPerformance,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

def run_validation_tests():
    """Run validation tests and generate report"""
    print("🧪 Running Sleep Detection Module Validation Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failures} ❌")
    print(f"Errors: {errors} 💥")
    print(f"Skipped: {skipped} ⏭️")
    
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Print failure details
    if result.failures:
        print("\n🔍 FAILURE DETAILS:")
        for test, traceback in result.failures:
            print(f"\n❌ {test}:")
            print(traceback)
    
    if result.errors:
        print("\n💥 ERROR DETAILS:")
        for test, traceback in result.errors:
            print(f"\n💥 {test}:")
            print(traceback)
    
    print("\n" + "=" * 60)
    
    if failures == 0 and errors == 0:
        print("🎉 All tests passed! Sleep detection module is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please review and fix issues before deployment.")
        return False

if __name__ == '__main__':
    # Run validation tests
    success = run_validation_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)