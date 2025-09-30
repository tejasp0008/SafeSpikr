import time
import numpy as np
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sleep_config import SleepDetectionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SleepMetrics:
    """Container for sleep detection metrics"""
    eye_closure_duration: float = 0.0
    blink_rate: float = 0.0
    head_movement_angle: float = 0.0
    drowsiness_score: float = 0.0
    distraction_score: float = 0.0
    eye_aspect_ratio: float = 0.3
    head_stability: float = 1.0
    attention_score: float = 1.0

@dataclass
class DetectionResult:
    """Container for detection results"""
    state: str = 'normal'  # 'sleeping', 'drowsy', 'distracted', 'normal'
    confidence: float = 0.0
    metrics: SleepMetrics = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = SleepMetrics()
        if self.timestamp is None:
            self.timestamp = datetime.now()

class TemporalSmoother:
    """Temporal smoothing for detection results"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: Deque[DetectionResult] = deque(maxlen=window_size)
    
    def add_result(self, result: DetectionResult):
        """Add a new detection result"""
        self.history.append(result)
    
    def get_smoothed_state(self) -> str:
        """Get smoothed state based on recent history"""
        if not self.history:
            return 'normal'
        
        # Count occurrences of each state
        state_counts = {}
        for result in self.history:
            state = result.state
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Return most frequent state
        return max(state_counts, key=state_counts.get)
    
    def get_smoothed_confidence(self) -> float:
        """Get average confidence from recent history"""
        if not self.history:
            return 0.0
        
        confidences = [result.confidence for result in self.history]
        return sum(confidences) / len(confidences)

class BlinkTracker:
    """Track blink patterns for drowsiness detection"""
    
    def __init__(self, blink_threshold: float = 0.20):
        self.blink_threshold = blink_threshold
        self.blink_times: List[datetime] = []
        self.eye_closed_start: Optional[datetime] = None
        self.is_eye_closed = False
        self.total_closure_time = 0.0
    
    def update(self, eye_aspect_ratio: float) -> Dict[str, float]:
        """Update blink tracking with new eye aspect ratio"""
        current_time = datetime.now()
        
        # Determine if eyes are closed
        eyes_closed = eye_aspect_ratio < self.blink_threshold
        
        # Track blink events
        if eyes_closed and not self.is_eye_closed:
            # Eyes just closed
            self.eye_closed_start = current_time
            self.is_eye_closed = True
            
        elif not eyes_closed and self.is_eye_closed:
            # Eyes just opened - record blink
            if self.eye_closed_start:
                closure_duration = (current_time - self.eye_closed_start).total_seconds()
                
                # If closure was brief (< 0.5s), count as blink
                if closure_duration < 0.5:
                    self.blink_times.append(current_time)
                else:
                    # Longer closure - add to total closure time
                    self.total_closure_time += closure_duration
                
            self.is_eye_closed = False
            self.eye_closed_start = None
        
        elif eyes_closed and self.is_eye_closed and self.eye_closed_start:
            # Eyes still closed - update total closure time
            closure_duration = (current_time - self.eye_closed_start).total_seconds()
            if closure_duration > 0.5:  # Only count extended closures
                self.total_closure_time += 0.1  # Add small increment
        
        # Clean old blink records (keep last minute)
        cutoff_time = current_time - timedelta(minutes=1)
        self.blink_times = [t for t in self.blink_times if t > cutoff_time]
        
        # Calculate metrics
        blink_rate = len(self.blink_times)  # Blinks per minute
        current_closure = 0.0
        
        if self.is_eye_closed and self.eye_closed_start:
            current_closure = (current_time - self.eye_closed_start).total_seconds()
        
        return {
            'blink_rate': blink_rate,
            'eye_closure_duration': current_closure,
            'total_closure_time': self.total_closure_time
        }

class HeadPoseTracker:
    """Track head pose and movement for distraction detection"""
    
    def __init__(self, stability_threshold: float = 5.0):
        self.stability_threshold = stability_threshold
        self.pose_history: Deque[Dict[str, float]] = deque(maxlen=30)  # 30 frames of history
        self.distraction_start: Optional[datetime] = None
        self.is_distracted = False
    
    def update(self, pose_data: Dict[str, float], movement_magnitude: float = 0.0) -> Dict[str, float]:
        """Update head pose tracking"""
        current_time = datetime.now()
        
        # Add current pose to history
        pose_entry = {
            'yaw': pose_data.get('Yaw', 0.0),
            'pitch': pose_data.get('Pitch', 0.0),
            'roll': pose_data.get('Roll', 0.0),
            'movement': movement_magnitude,
            'timestamp': current_time
        }
        self.pose_history.append(pose_entry)
        
        # Calculate head stability
        if len(self.pose_history) > 5:
            recent_poses = list(self.pose_history)[-5:]
            
            # Calculate variance in head angles
            yaw_values = [p['yaw'] for p in recent_poses]
            pitch_values = [p['pitch'] for p in recent_poses]
            
            yaw_variance = np.var(yaw_values) if len(yaw_values) > 1 else 0.0
            pitch_variance = np.var(pitch_values) if len(pitch_values) > 1 else 0.0
            
            # Calculate stability score (lower variance = higher stability)
            stability = max(0.0, 1.0 - (yaw_variance + pitch_variance) / 100.0)
        else:
            stability = 1.0
        
        # Check for distraction (head turned away significantly)
        current_yaw = abs(pose_data.get('Yaw', 0.0))
        current_pitch = abs(pose_data.get('Pitch', 0.0))
        
        is_head_turned = (current_yaw > self.stability_threshold or 
                         current_pitch > self.stability_threshold)
        
        # Track distraction duration
        distraction_duration = 0.0
        if is_head_turned and not self.is_distracted:
            self.distraction_start = current_time
            self.is_distracted = True
        elif not is_head_turned and self.is_distracted:
            self.is_distracted = False
            self.distraction_start = None
        elif is_head_turned and self.is_distracted and self.distraction_start:
            distraction_duration = (current_time - self.distraction_start).total_seconds()
        
        return {
            'head_stability': stability,
            'distraction_duration': distraction_duration,
            'head_angle': max(current_yaw, current_pitch),
            'is_distracted': self.is_distracted
        }

class SleepDetectionEngine:
    """Core engine for sleep and distraction detection"""
    
    def __init__(self):
        self.config = SleepDetectionConfig()
        self.blink_tracker = BlinkTracker()
        self.head_tracker = HeadPoseTracker()
        self.temporal_smoother = TemporalSmoother()
        
        # Detection history
        self.detection_history: List[DetectionResult] = []
        self.last_analysis_time = datetime.now()
        
        logger.info("Sleep Detection Engine initialized")
    
    def analyze_frame(self, landmarks: Any, emotions: Any, pose_data: Dict[str, float], 
                     eye_aspect_ratio: float, movement_magnitude: float = 0.0) -> DetectionResult:
        """Analyze a single frame for sleep/distraction detection"""
        
        current_time = datetime.now()
        
        # Update trackers
        blink_metrics = self.blink_tracker.update(eye_aspect_ratio)
        head_metrics = self.head_tracker.update(pose_data, movement_magnitude)
        
        # Calculate comprehensive metrics
        metrics = SleepMetrics(
            eye_closure_duration=blink_metrics['eye_closure_duration'],
            blink_rate=blink_metrics['blink_rate'],
            head_movement_angle=head_metrics['head_angle'],
            eye_aspect_ratio=eye_aspect_ratio,
            head_stability=head_metrics['head_stability'],
            drowsiness_score=self._calculate_drowsiness_score(blink_metrics, eye_aspect_ratio),
            distraction_score=self._calculate_distraction_score(head_metrics),
            attention_score=self._calculate_attention_score(blink_metrics, head_metrics)
        )
        
        # Determine state
        state, confidence = self._classify_state(metrics, head_metrics)
        
        # Create result
        result = DetectionResult(
            state=state,
            confidence=confidence,
            metrics=metrics,
            timestamp=current_time
        )
        
        # Add to temporal smoother
        self.temporal_smoother.add_result(result)
        
        # Store in history
        self.detection_history.append(result)
        
        # Keep only recent history (last 5 minutes)
        cutoff_time = current_time - timedelta(minutes=5)
        self.detection_history = [r for r in self.detection_history if r.timestamp > cutoff_time]
        
        return result
    
    def _calculate_drowsiness_score(self, blink_metrics: Dict[str, float], eye_aspect_ratio: float) -> float:
        """Calculate drowsiness score based on eye behavior"""
        score = 0.0
        
        # Factor 1: Eye closure duration
        closure_duration = blink_metrics['eye_closure_duration']
        if closure_duration > 1.0:
            score += min(50.0, closure_duration * 10)  # Up to 50 points for closure
        
        # Factor 2: Blink rate (too high or too low indicates drowsiness)
        blink_rate = blink_metrics['blink_rate']
        if blink_rate > self.config.DROWSY_BLINK_RATE_THRESHOLD:
            score += 30.0  # High blink rate
        elif blink_rate < 5:  # Very low blink rate
            score += 20.0
        
        # Factor 3: Eye aspect ratio (lower = more closed)
        if eye_aspect_ratio < 0.2:
            score += 40.0
        elif eye_aspect_ratio < 0.25:
            score += 20.0
        
        return min(100.0, score)
    
    def _calculate_distraction_score(self, head_metrics: Dict[str, float]) -> float:
        """Calculate distraction score based on head movement"""
        score = 0.0
        
        # Factor 1: Head angle deviation
        head_angle = head_metrics['head_angle']
        if head_angle > self.config.DISTRACTION_HEAD_ANGLE_THRESHOLD:
            score += min(50.0, head_angle * 2)
        
        # Factor 2: Distraction duration
        distraction_duration = head_metrics['distraction_duration']
        if distraction_duration > self.config.DISTRACTION_DURATION_THRESHOLD:
            score += min(40.0, distraction_duration * 5)
        
        # Factor 3: Head stability (lower stability = higher distraction)
        stability = head_metrics['head_stability']
        if stability < 0.5:
            score += (1.0 - stability) * 30.0
        
        return min(100.0, score)
    
    def _calculate_attention_score(self, blink_metrics: Dict[str, float], head_metrics: Dict[str, float]) -> float:
        """Calculate overall attention score"""
        # Start with perfect attention
        attention = 100.0
        
        # Reduce based on drowsiness indicators
        if blink_metrics['eye_closure_duration'] > 2.0:
            attention -= 50.0
        
        # Reduce based on distraction indicators
        if head_metrics['distraction_duration'] > 3.0:
            attention -= 40.0
        
        # Reduce based on head stability
        attention *= head_metrics['head_stability']
        
        return max(0.0, attention)
    
    def _classify_state(self, metrics: SleepMetrics, head_metrics: Dict[str, float]) -> tuple[str, float]:
        """Classify the current state based on metrics"""
        
        # Check for sleeping (highest priority)
        if metrics.eye_closure_duration >= self.config.SLEEP_EYE_CLOSURE_THRESHOLD:
            confidence = min(95.0, 70.0 + metrics.eye_closure_duration * 5)
            return 'sleeping', confidence
        
        # Check for distraction
        if (head_metrics['distraction_duration'] >= self.config.DISTRACTION_DURATION_THRESHOLD or
            metrics.distraction_score > 60.0):
            confidence = min(90.0, 60.0 + metrics.distraction_score * 0.3)
            return 'distracted', confidence
        
        # Check for drowsiness
        if (metrics.drowsiness_score > 50.0 or 
            metrics.blink_rate > self.config.DROWSY_BLINK_RATE_THRESHOLD):
            confidence = min(85.0, 50.0 + metrics.drowsiness_score * 0.4)
            return 'drowsy', confidence
        
        # Default to normal state
        confidence = metrics.attention_score * 0.8  # Scale attention score to confidence
        return 'normal', max(50.0, confidence)
    
    def get_smoothed_result(self) -> DetectionResult:
        """Get temporally smoothed detection result"""
        if not self.detection_history:
            return DetectionResult()
        
        # Get the most recent raw result
        latest_result = self.detection_history[-1]
        
        # Apply temporal smoothing
        smoothed_state = self.temporal_smoother.get_smoothed_state()
        smoothed_confidence = self.temporal_smoother.get_smoothed_confidence()
        
        # Create smoothed result
        return DetectionResult(
            state=smoothed_state,
            confidence=smoothed_confidence,
            metrics=latest_result.metrics,  # Use latest metrics
            timestamp=latest_result.timestamp
        )
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of recent detection activity"""
        if not self.detection_history:
            return {'total_detections': 0}
        
        recent_history = self.detection_history[-60:]  # Last 60 detections
        
        # Count states
        state_counts = {}
        total_confidence = 0.0
        
        for result in recent_history:
            state = result.state
            state_counts[state] = state_counts.get(state, 0) + 1
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(recent_history) if recent_history else 0.0
        
        return {
            'total_detections': len(recent_history),
            'state_distribution': state_counts,
            'average_confidence': avg_confidence,
            'current_state': recent_history[-1].state if recent_history else 'unknown',
            'detection_rate': len(self.detection_history) / max(1, (datetime.now() - self.last_analysis_time).total_seconds())
        }
    
    def reset_tracking(self):
        """Reset all tracking state"""
        self.blink_tracker = BlinkTracker()
        self.head_tracker = HeadPoseTracker()
        self.temporal_smoother = TemporalSmoother()
        self.detection_history.clear()
        logger.info("Sleep detection tracking reset")