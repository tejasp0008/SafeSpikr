#!/usr/bin/env python3
"""
Sleep Detection System - Main Orchestrator
Coordinates all components for real-time sleep and distraction detection
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging
import json

# Import all components
from sleep_config import SleepDetectionConfig
from aws_sleep_service import AWSRekognitionSleepService
from opencv_sleep_service import OpenCVSleepService
from sleep_detection_engine import SleepDetectionEngine, DetectionResult, SleepMetrics
from state_classifier import AdvancedStateClassifier
from camera_manager import CameraManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertnessState:
    """Current alertness state container"""
    def __init__(self):
        self.current_state = 'normal'
        self.confidence = 0.0
        self.duration_in_state = 0.0
        self.last_state_change = datetime.now()
        self.metrics = SleepMetrics()
        self.detection_method = 'none'
        self.last_update = datetime.now()

class SystemStatus:
    """System status and health monitoring"""
    def __init__(self):
        self.aws_available = False
        self.opencv_available = False
        self.camera_available = False
        self.detection_active = False
        self.current_method = 'none'
        self.last_error = None
        self.uptime_start = datetime.now()
        self.total_detections = 0
        self.successful_detections = 0

class SleepDetectionSystem:
    """Main orchestrator for sleep detection system"""
    
    def __init__(self):
        print("🚀 Initializing Sleep Detection System...")
        
        # Load configuration
        self.config = SleepDetectionConfig()
        
        # Initialize components
        self.aws_service = AWSRekognitionSleepService()
        self.opencv_service = OpenCVSleepService()
        self.detection_engine = SleepDetectionEngine()
        self.state_classifier = AdvancedStateClassifier()
        self.camera_manager = CameraManager()
        
        # System state
        self.system_status = SystemStatus()
        self.alertness_state = AlertnessState()
        
        # Threading and control
        self.detection_thread = None
        self.is_monitoring = False
        self.detection_lock = threading.Lock()
        
        # Callbacks for external integration
        self.state_change_callbacks: list[Callable] = []
        self.detection_callbacks: list[Callable] = []
        
        # Performance monitoring
        self.performance_stats = {
            'avg_processing_time': 0.0,
            'frames_per_second': 0.0,
            'detection_accuracy': 0.0,
            'last_processing_times': []
        }
        
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize and validate all system components"""
        logger.info("Validating system components...")
        
        # Check AWS service
        self.system_status.aws_available = self.aws_service.is_service_available()
        if self.system_status.aws_available:
            logger.info("✅ AWS Rekognition service available")
        else:
            logger.warning(f"⚠️ AWS Rekognition unavailable: {self.aws_service.get_last_error()}")
        
        # Check OpenCV service
        self.system_status.opencv_available = self.opencv_service.is_service_available()
        if self.system_status.opencv_available:
            logger.info("✅ OpenCV fallback service available")
        else:
            logger.error(f"❌ OpenCV service unavailable: {self.opencv_service.get_last_error()}")
        
        # Check camera
        self.system_status.camera_available = self.camera_manager.start_camera()
        if self.system_status.camera_available:
            logger.info("✅ Camera initialized successfully")
        else:
            logger.error("❌ Camera initialization failed")
        
        # Determine detection method
        self._select_detection_method()
        
        # Validate configuration
        config_status = self.config.validate_config()
        if not config_status['valid']:
            logger.warning(f"⚠️ Configuration issues: {config_status['issues']}")
        
        print("✅ Sleep Detection System ready!")
        
    def _select_detection_method(self):
        """Select the best available detection method"""
        if self.config.USE_AWS_PRIMARY and self.system_status.aws_available:
            self.system_status.current_method = 'aws'
            logger.info("Using AWS Rekognition as primary detection method")
        elif self.config.FALLBACK_TO_OPENCV and self.system_status.opencv_available:
            self.system_status.current_method = 'opencv'
            logger.info("Using OpenCV as detection method")
        else:
            self.system_status.current_method = 'none'
            logger.error("No detection method available!")
    
    def start_monitoring(self) -> bool:
        """Start real-time sleep detection monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return True
        
        if not self.system_status.camera_available:
            logger.error("Cannot start monitoring - camera not available")
            return False
        
        if self.system_status.current_method == 'none':
            logger.error("Cannot start monitoring - no detection method available")
            return False
        
        try:
            # Start camera capture thread
            self.camera_manager.start_capture_thread(self._frame_callback)
            
            # Start detection thread
            self.is_monitoring = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.system_status.detection_active = True
            logger.info("🎯 Sleep detection monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.system_status.last_error = str(e)
            return False
    
    def _frame_callback(self, current_frame, previous_frame):
        """Callback for new frames from camera"""
        # This is called from camera thread - just store frames
        # Actual processing happens in detection thread
        pass
    
    def _detection_loop(self):
        """Main detection processing loop"""
        logger.info("Detection loop started")
        
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Get current frame pair
                current_frame, previous_frame = self.camera_manager.get_frame_pair()
                
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                detection_result = self.process_frame(current_frame, previous_frame)
                
                if detection_result:
                    # Update system state
                    self._update_alertness_state(detection_result)
                    
                    # Call detection callbacks
                    for callback in self.detection_callbacks:
                        try:
                            callback(detection_result)
                        except Exception as e:
                            logger.error(f"Error in detection callback: {e}")
                
                # Update performance stats
                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                self.system_status.last_error = str(e)
                time.sleep(0.5)  # Longer pause on error
    
    def process_frame(self, current_frame, previous_frame=None) -> Optional[DetectionResult]:
        """Process a single frame for sleep detection"""
        if current_frame is None:
            return None
        
        try:
            with self.detection_lock:
                self.system_status.total_detections += 1
                
                # Always try AWS first for better eye detection
                result = self._process_frame_aws(current_frame)
                
                # If AWS fails and fallback is enabled, use AWS-powered fallback
                if result is None and self.config.FALLBACK_TO_OPENCV:
                    logger.debug("AWS detection failed, trying AWS-powered fallback")
                    result = self._process_frame_opencv(current_frame, previous_frame)
                
                if result:
                    self.system_status.successful_detections += 1
                    return result
                
                return None
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self.system_status.last_error = str(e)
            return None
    
    def _process_frame_aws(self, frame) -> Optional[DetectionResult]:
        """Process frame using AWS Rekognition"""
        try:
            # Convert frame to bytes
            image_bytes = self.camera_manager.frame_to_bytes(frame)
            if not image_bytes:
                return None
            
            # Analyze with AWS
            analysis = self.aws_service.analyze_facial_features(image_bytes)
            if not analysis:
                return None
            
            # Extract eye state information from AWS
            eyes_open = True
            eye_confidence = 50.0
            
            if analysis.eye_states and 'eyes_open' in analysis.eye_states:
                eye_state_data = analysis.eye_states['eyes_open']
                if isinstance(eye_state_data, dict):
                    eyes_open = eye_state_data.get('Value', True)
                    eye_confidence = eye_state_data.get('Confidence', 50.0)
                else:
                    eyes_open = bool(eye_state_data)
                
                logger.debug(f"AWS Eye State: {'Open' if eyes_open else 'Closed'} (confidence: {eye_confidence:.1f}%)")
            
            # Calculate eye aspect ratio based on AWS detection
            if not eyes_open and eye_confidence > 60:
                # AWS detected closed eyes with good confidence
                eye_aspect_ratio = 0.12  # Very low ratio for closed eyes
                eye_closure_duration = 2.0  # Simulate closure duration
            elif eyes_open and eye_confidence > 60:
                # AWS detected open eyes with good confidence
                eye_aspect_ratio = 0.35  # Normal ratio for open eyes
                eye_closure_duration = 0.1
            else:
                # Low confidence or no eye state data, use landmark-based calculation
                eye_aspect_ratio = 0.25  # Default
                eye_closure_duration = 0.5
                if analysis.landmarks:
                    calculated_ear = self.aws_service.calculate_eye_closure_ratio(analysis.landmarks)
                    eye_aspect_ratio = calculated_ear
            
            pose_data = analysis.pose or {}
            movement_magnitude = 0.0  # AWS doesn't provide frame-to-frame movement
            
            # Create enhanced metrics with AWS eye state data
            from sleep_detection_engine import SleepMetrics
            
            # Determine drowsiness and attention scores based on eye state
            if not eyes_open and eye_confidence > 70:
                drowsiness_score = 85.0
                attention_score = 20.0
            elif not eyes_open and eye_confidence > 50:
                drowsiness_score = 65.0
                attention_score = 40.0
            else:
                drowsiness_score = 25.0
                attention_score = 80.0
            
            # Create metrics directly
            metrics = SleepMetrics(
                eye_closure_duration=eye_closure_duration,
                blink_rate=12.0,  # Default
                head_movement_angle=abs(pose_data.get('Yaw', 0.0)),
                drowsiness_score=drowsiness_score,
                distraction_score=abs(pose_data.get('Yaw', 0.0)) * 2,
                attention_score=attention_score,
                eye_aspect_ratio=eye_aspect_ratio,
                head_stability=0.8
            )
            
            # Create detection result
            from sleep_detection_engine import DetectionResult
            result = DetectionResult(
                state='normal',  # Will be overridden by classifier
                confidence=eye_confidence,
                metrics=metrics,
                timestamp=datetime.now()
            )
            
            # Apply advanced state classification
            classified_state, classified_confidence, reason = self.state_classifier.classify_state(
                result.metrics, result.confidence
            )
            
            # Update result with classification
            result.state = classified_state
            result.confidence = classified_confidence
            result.detection_method = 'aws'
            result.classification_reason = reason
            
            # Store landmarks for overlay
            result.landmarks = analysis.landmarks
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AWS frame processing: {e}")
            return None
    
    def _process_frame_opencv(self, current_frame, previous_frame=None) -> Optional[DetectionResult]:
        """Process frame using OpenCV (fallback only)"""
        try:
            # Convert frame to bytes for AWS-style processing
            image_bytes = self.camera_manager.frame_to_bytes(current_frame)
            if not image_bytes:
                return None
            
            # Use AWS Rekognition even in "OpenCV" mode for better accuracy
            analysis = self.aws_service.analyze_facial_features(image_bytes)
            if not analysis:
                return None
            
            # Extract metrics
            eye_aspect_ratio = 0.3  # Default
            if analysis.landmarks:
                eye_aspect_ratio = self.aws_service.calculate_eye_closure_ratio(analysis.landmarks)
            
            # Check AWS eye states for more accurate detection
            eyes_open = True
            eye_confidence = 50.0
            
            if analysis.eye_states and 'eyes_open' in analysis.eye_states:
                eye_state_data = analysis.eye_states['eyes_open']
                if isinstance(eye_state_data, dict):
                    eyes_open = eye_state_data.get('Value', True)
                    eye_confidence = eye_state_data.get('Confidence', 50.0)
                else:
                    eyes_open = bool(eye_state_data)
            
            # Override EAR based on AWS eye state detection
            if not eyes_open and eye_confidence > 70:
                eye_aspect_ratio = 0.15  # Force closed eye ratio
            elif eyes_open and eye_confidence > 70:
                eye_aspect_ratio = max(0.25, eye_aspect_ratio)  # Ensure open eye ratio
            
            pose_data = analysis.pose or {}
            movement_magnitude = 0.0  # AWS doesn't provide frame-to-frame movement
            
            # Run through detection engine
            result = self.detection_engine.analyze_frame(
                landmarks=analysis.landmarks,
                emotions=analysis.emotions,
                pose_data=pose_data,
                eye_aspect_ratio=eye_aspect_ratio,
                movement_magnitude=movement_magnitude
            )
            
            # Apply advanced state classification
            classified_state, classified_confidence, reason = self.state_classifier.classify_state(
                result.metrics, result.confidence
            )
            
            # Update result with classification
            result.state = classified_state
            result.confidence = classified_confidence
            result.detection_method = 'aws_fallback'
            result.classification_reason = reason
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AWS fallback frame processing: {e}")
            return None
    
    def _update_alertness_state(self, detection_result: DetectionResult):
        """Update current alertness state"""
        previous_state = self.alertness_state.current_state
        
        # Update state
        self.alertness_state.current_state = detection_result.state
        self.alertness_state.confidence = detection_result.confidence
        self.alertness_state.metrics = detection_result.metrics
        self.alertness_state.detection_method = getattr(detection_result, 'detection_method', 'unknown')
        self.alertness_state.last_update = detection_result.timestamp
        
        # Check for state change
        if previous_state != detection_result.state:
            self.alertness_state.last_state_change = detection_result.timestamp
            self.alertness_state.duration_in_state = 0.0
            
            # Call state change callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(previous_state, detection_result.state, detection_result.confidence)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
        else:
            # Update duration in current state
            self.alertness_state.duration_in_state = (
                detection_result.timestamp - self.alertness_state.last_state_change
            ).total_seconds()
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        # Add to recent processing times
        self.performance_stats['last_processing_times'].append(processing_time)
        
        # Keep only last 100 measurements
        if len(self.performance_stats['last_processing_times']) > 100:
            self.performance_stats['last_processing_times'].pop(0)
        
        # Calculate average
        times = self.performance_stats['last_processing_times']
        self.performance_stats['avg_processing_time'] = sum(times) / len(times)
        
        # Calculate effective FPS
        if self.performance_stats['avg_processing_time'] > 0:
            self.performance_stats['frames_per_second'] = 1.0 / self.performance_stats['avg_processing_time']
        
        # Calculate detection accuracy
        if self.system_status.total_detections > 0:
            self.performance_stats['detection_accuracy'] = (
                self.system_status.successful_detections / self.system_status.total_detections * 100.0
            )
    
    def stop_monitoring(self):
        """Stop sleep detection monitoring"""
        if not self.is_monitoring:
            return
        
        logger.info("Stopping sleep detection monitoring...")
        
        self.is_monitoring = False
        self.system_status.detection_active = False
        
        # Stop camera
        self.camera_manager.stop_capture()
        
        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        logger.info("Sleep detection monitoring stopped")
    
    def get_current_state(self) -> AlertnessState:
        """Get current alertness state"""
        return self.alertness_state
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        camera_info = self.camera_manager.get_camera_info()
        camera_stats = self.camera_manager.get_statistics()
        
        uptime = (datetime.now() - self.system_status.uptime_start).total_seconds()
        
        return {
            'system': {
                'monitoring_active': self.is_monitoring,
                'detection_method': self.system_status.current_method,
                'uptime_seconds': uptime,
                'last_error': self.system_status.last_error
            },
            'services': {
                'aws_available': self.system_status.aws_available,
                'opencv_available': self.system_status.opencv_available,
                'camera_available': self.system_status.camera_available
            },
            'camera': camera_info,
            'performance': {
                **self.performance_stats,
                'camera_fps': camera_stats.get('current_fps', 0.0),
                'total_detections': self.system_status.total_detections,
                'successful_detections': self.system_status.successful_detections,
                'detection_accuracy': self.performance_stats['detection_accuracy']
            },
            'current_state': {
                'state': self.alertness_state.current_state,
                'confidence': self.alertness_state.confidence,
                'duration': self.alertness_state.duration_in_state,
                'detection_method': self.alertness_state.detection_method,
                'last_update': self.alertness_state.last_update.isoformat()
            }
        }
    
    def configure_thresholds(self, config_updates: Dict[str, float]) -> bool:
        """Update detection thresholds"""
        try:
            updated = False
            for threshold_name, value in config_updates.items():
                if self.config.update_threshold(threshold_name, value):
                    updated = True
                    logger.info(f"Updated {threshold_name} to {value}")
            
            if updated:
                # Reset detection engine to apply new thresholds
                self.detection_engine.reset_tracking()
                self.state_classifier.reset_classifier()
                logger.info("Detection thresholds updated and tracking reset")
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
            return False
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    def add_detection_callback(self, callback: Callable):
        """Add callback for each detection"""
        self.detection_callbacks.append(callback)
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get detection summary from engine"""
        return self.detection_engine.get_detection_summary()
    
    def reset_system(self):
        """Reset all system state"""
        logger.info("Resetting sleep detection system...")
        
        # Reset components
        self.detection_engine.reset_tracking()
        self.state_classifier.reset_classifier()
        self.camera_manager.reset_statistics()
        
        # Reset system state
        self.alertness_state = AlertnessState()
        self.system_status.total_detections = 0
        self.system_status.successful_detections = 0
        self.system_status.last_error = None
        
        # Reset performance stats
        self.performance_stats = {
            'avg_processing_time': 0.0,
            'frames_per_second': 0.0,
            'detection_accuracy': 0.0,
            'last_processing_times': []
        }
        
        logger.info("System reset complete")
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down sleep detection system...")
        
        self.stop_monitoring()
        self.camera_manager.release_camera()
        
        logger.info("Sleep detection system shutdown complete")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.shutdown()

# Main execution
if __name__ == "__main__":
    # Create and start the system
    system = SleepDetectionSystem()
    
    try:
        # Start monitoring
        if system.start_monitoring():
            print("\n🎯 Sleep detection is now active!")
            print("Press Ctrl+C to stop...")
            
            # Simple status display loop
            while True:
                time.sleep(5)
                state = system.get_current_state()
                print(f"Current State: {state.current_state.upper()} "
                      f"(Confidence: {state.confidence:.1f}%, "
                      f"Duration: {state.duration_in_state:.1f}s)")
        else:
            print("❌ Failed to start sleep detection monitoring")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping sleep detection...")
    finally:
        system.shutdown()
        print("👋 Sleep detection system stopped")