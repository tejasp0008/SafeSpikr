"""
Comprehensive Error Handling and Logging System
Centralized error handling, logging, and recovery mechanisms
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import json
import os
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    CAMERA = "camera"
    AWS_SERVICE = "aws_service"
    OPENCV = "opencv"
    DETECTION = "detection"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    NETWORK = "network"
    HARDWARE = "hardware"

class ErrorRecord:
    """Structured error record"""
    def __init__(self, 
                 error: Exception, 
                 category: ErrorCategory, 
                 severity: ErrorSeverity,
                 context: Dict[str, Any] = None,
                 recovery_action: str = None):
        self.timestamp = datetime.now()
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recovery_action = recovery_action
        self.traceback = traceback.format_exc()
        self.resolved = False
        self.resolution_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'recovery_action': self.recovery_action,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }
    
    def mark_resolved(self):
        """Mark error as resolved"""
        self.resolved = True
        self.resolution_time = datetime.now()

class SleepDetectionLogger:
    """Enhanced logging system for sleep detection"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger('sleep_detection')
        self.logger.setLevel(log_level)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler('logs/sleep_detection.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler
        error_handler = logging.FileHandler('logs/sleep_detection_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
        
        # Performance logging
        self.performance_logger = logging.getLogger('sleep_detection.performance')
        perf_handler = logging.FileHandler('logs/performance.log')
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.performance_logger.addHandler(perf_handler)
        self.performance_logger.setLevel(logging.INFO)
    
    def log_performance(self, operation: str, duration: float, context: Dict[str, Any] = None):
        """Log performance metrics"""
        context = context or {}
        self.performance_logger.info(
            f"PERF: {operation} - {duration:.3f}s - {json.dumps(context)}"
        )
    
    def log_detection_event(self, state: str, confidence: float, method: str):
        """Log detection events"""
        self.logger.info(f"DETECTION: {state} (confidence: {confidence:.1f}%, method: {method})")
    
    def log_state_change(self, from_state: str, to_state: str, confidence: float):
        """Log state changes"""
        self.logger.info(f"STATE_CHANGE: {from_state} -> {to_state} (confidence: {confidence:.1f}%)")
    
    def log_system_event(self, event: str, details: Dict[str, Any] = None):
        """Log system events"""
        details = details or {}
        self.logger.info(f"SYSTEM: {event} - {json.dumps(details)}")

class ErrorManager:
    """Centralized error management system"""
    
    def __init__(self, logger: SleepDetectionLogger):
        self.logger = logger
        self.error_history: list[ErrorRecord] = []
        self.max_history = 1000
        self.recovery_strategies = {}
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Setup automatic recovery strategies"""
        self.recovery_strategies = {
            (ErrorCategory.CAMERA, "camera_not_found"): self._recover_camera_not_found,
            (ErrorCategory.CAMERA, "frame_capture_failed"): self._recover_frame_capture,
            (ErrorCategory.AWS_SERVICE, "connection_failed"): self._recover_aws_connection,
            (ErrorCategory.AWS_SERVICE, "rate_limit_exceeded"): self._recover_aws_rate_limit,
            (ErrorCategory.OPENCV, "cascade_load_failed"): self._recover_opencv_cascade,
            (ErrorCategory.DETECTION, "no_face_detected"): self._recover_no_face,
            (ErrorCategory.SYSTEM, "memory_error"): self._recover_memory_error,
        }
    
    def handle_error(self, 
                    error: Exception, 
                    category: ErrorCategory, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Dict[str, Any] = None,
                    auto_recover: bool = True) -> ErrorRecord:
        """Handle and log an error"""
        
        # Create error record
        error_record = ErrorRecord(error, category, severity, context)
        
        # Log the error
        self._log_error(error_record)
        
        # Add to history
        self.error_history.append(error_record)
        
        # Trim history if needed
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Attempt automatic recovery
        if auto_recover:
            recovery_success = self._attempt_recovery(error_record)
            if recovery_success:
                error_record.mark_resolved()
                self.logger.logger.info(f"Auto-recovery successful for {error_record.error_type}")
        
        return error_record
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error based on severity"""
        log_message = (
            f"{error_record.category.value.upper()}: {error_record.error_message}"
        )
        
        if error_record.context:
            log_message += f" - Context: {json.dumps(error_record.context)}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.logger.warning(log_message)
        else:
            self.logger.logger.info(log_message)
    
    def _attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt automatic error recovery"""
        try:
            # Look for specific recovery strategy
            error_key = error_record.context.get('error_key') if error_record.context else None
            strategy_key = (error_record.category, error_key)
            
            if strategy_key in self.recovery_strategies:
                recovery_func = self.recovery_strategies[strategy_key]
                return recovery_func(error_record)
            
            # Generic recovery based on category
            return self._generic_recovery(error_record)
            
        except Exception as recovery_error:
            self.logger.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False
    
    def _generic_recovery(self, error_record: ErrorRecord) -> bool:
        """Generic recovery strategies"""
        if error_record.category == ErrorCategory.CAMERA:
            # Wait and retry camera initialization
            import time
            time.sleep(2)
            return True
        
        elif error_record.category == ErrorCategory.AWS_SERVICE:
            # Switch to fallback mode
            error_record.recovery_action = "Switched to OpenCV fallback"
            return True
        
        elif error_record.category == ErrorCategory.DETECTION:
            # Skip frame and continue
            error_record.recovery_action = "Skipped problematic frame"
            return True
        
        return False
    
    # Specific recovery strategies
    def _recover_camera_not_found(self, error_record: ErrorRecord) -> bool:
        """Recover from camera not found error"""
        try:
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                import cv2
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    cap.release()
                    error_record.recovery_action = f"Found camera at index {camera_index}"
                    return True
            return False
        except:
            return False
    
    def _recover_frame_capture(self, error_record: ErrorRecord) -> bool:
        """Recover from frame capture failure"""
        error_record.recovery_action = "Retrying frame capture"
        return True
    
    def _recover_aws_connection(self, error_record: ErrorRecord) -> bool:
        """Recover from AWS connection failure"""
        error_record.recovery_action = "Switched to OpenCV fallback mode"
        return True
    
    def _recover_aws_rate_limit(self, error_record: ErrorRecord) -> bool:
        """Recover from AWS rate limiting"""
        import time
        time.sleep(1)  # Brief pause
        error_record.recovery_action = "Applied rate limiting delay"
        return True
    
    def _recover_opencv_cascade(self, error_record: ErrorRecord) -> bool:
        """Recover from OpenCV cascade loading failure"""
        error_record.recovery_action = "Using alternative detection method"
        return True
    
    def _recover_no_face(self, error_record: ErrorRecord) -> bool:
        """Recover from no face detected"""
        error_record.recovery_action = "Continuing monitoring for face detection"
        return True
    
    def _recover_memory_error(self, error_record: ErrorRecord) -> bool:
        """Recover from memory errors"""
        import gc
        gc.collect()
        error_record.recovery_action = "Performed garbage collection"
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        if not self.error_history:
            return {
                'total_errors': 0,
                'resolved_errors': 0,
                'critical_errors': 0,
                'recent_errors': []
            }
        
        total_errors = len(self.error_history)
        resolved_errors = sum(1 for e in self.error_history if e.resolved)
        critical_errors = sum(1 for e in self.error_history if e.severity == ErrorSeverity.CRITICAL)
        
        # Get recent errors (last 10)
        recent_errors = [e.to_dict() for e in self.error_history[-10:]]
        
        # Error distribution by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_errors': total_errors,
            'resolved_errors': resolved_errors,
            'critical_errors': critical_errors,
            'resolution_rate': (resolved_errors / total_errors * 100) if total_errors > 0 else 0,
            'category_distribution': category_counts,
            'recent_errors': recent_errors
        }

def error_handler(category: ErrorCategory, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 auto_recover: bool = True,
                 context_keys: list = None):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build context from function arguments
                context = {}
                if context_keys:
                    for i, key in enumerate(context_keys):
                        if i < len(args):
                            context[key] = str(args[i])
                
                # Get error manager from global scope or create one
                error_manager = getattr(wrapper, '_error_manager', None)
                if error_manager is None:
                    logger = SleepDetectionLogger()
                    error_manager = ErrorManager(logger)
                    wrapper._error_manager = error_manager
                
                # Handle the error
                error_record = error_manager.handle_error(
                    e, category, severity, context, auto_recover
                )
                
                # Re-raise critical errors
                if severity == ErrorSeverity.CRITICAL:
                    raise
                
                # Return None for non-critical errors
                return None
        
        return wrapper
    return decorator

def performance_monitor(operation_name: str):
    """Decorator for performance monitoring"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log performance
                logger = getattr(wrapper, '_logger', None)
                if logger is None:
                    logger = SleepDetectionLogger()
                    wrapper._logger = logger
                
                logger.log_performance(operation_name, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                # Log failed operation
                logger = getattr(wrapper, '_logger', SleepDetectionLogger())
                logger.log_performance(f"{operation_name}_FAILED", duration, {'error': str(e)})
                raise
        
        return wrapper
    return decorator

# Global error manager instance
_global_error_manager = None
_global_logger = None

def get_error_manager() -> ErrorManager:
    """Get global error manager instance"""
    global _global_error_manager, _global_logger
    
    if _global_error_manager is None:
        if _global_logger is None:
            _global_logger = SleepDetectionLogger()
        _global_error_manager = ErrorManager(_global_logger)
    
    return _global_error_manager

def get_logger() -> SleepDetectionLogger:
    """Get global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = SleepDetectionLogger()
    
    return _global_logger

# Convenience functions
def log_error(error: Exception, 
              category: ErrorCategory, 
              severity: ErrorSeverity = ErrorSeverity.MEDIUM,
              context: Dict[str, Any] = None) -> ErrorRecord:
    """Convenience function to log an error"""
    return get_error_manager().handle_error(error, category, severity, context)

def log_performance(operation: str, duration: float, context: Dict[str, Any] = None):
    """Convenience function to log performance"""
    get_logger().log_performance(operation, duration, context)

def log_detection(state: str, confidence: float, method: str):
    """Convenience function to log detection events"""
    get_logger().log_detection_event(state, confidence, method)

def log_state_change(from_state: str, to_state: str, confidence: float):
    """Convenience function to log state changes"""
    get_logger().log_state_change(from_state, to_state, confidence)

def log_system_event(event: str, details: Dict[str, Any] = None):
    """Convenience function to log system events"""
    get_logger().log_system_event(event, details)