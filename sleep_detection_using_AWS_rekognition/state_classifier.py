from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging
from sleep_config import SleepDetectionConfig
from sleep_detection_engine import SleepMetrics, DetectionResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StateTransition:
    """Represents a state transition event"""
    from_state: str
    to_state: str
    timestamp: datetime
    confidence: float
    trigger_reason: str

class StateHistory:
    """Manages state history and transitions"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.transitions: deque[StateTransition] = deque(maxlen=max_history)
        self.current_state = 'normal'
        self.state_start_time = datetime.now()
        self.state_durations = {'normal': 0.0, 'drowsy': 0.0, 'sleeping': 0.0, 'distracted': 0.0}
    
    def add_transition(self, new_state: str, confidence: float, reason: str):
        """Add a state transition"""
        if new_state != self.current_state:
            # Record the transition
            transition = StateTransition(
                from_state=self.current_state,
                to_state=new_state,
                timestamp=datetime.now(),
                confidence=confidence,
                trigger_reason=reason
            )
            self.transitions.append(transition)
            
            # Update state duration
            duration = (datetime.now() - self.state_start_time).total_seconds()
            self.state_durations[self.current_state] += duration
            
            # Update current state
            self.current_state = new_state
            self.state_start_time = datetime.now()
            
            logger.info(f"State transition: {transition.from_state} -> {transition.to_state} ({reason})")
    
    def get_current_state_duration(self) -> float:
        """Get duration of current state in seconds"""
        return (datetime.now() - self.state_start_time).total_seconds()
    
    def get_recent_transitions(self, minutes: int = 5) -> List[StateTransition]:
        """Get transitions from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [t for t in self.transitions if t.timestamp > cutoff_time]

class HysteresisClassifier:
    """Classifier with hysteresis to prevent rapid state switching"""
    
    def __init__(self, config: SleepDetectionConfig):
        self.config = config
        self.state_thresholds = {
            'sleeping': {
                'enter': 0.8,  # High confidence required to enter sleep state
                'exit': 0.4    # Lower confidence required to exit sleep state
            },
            'drowsy': {
                'enter': 0.6,
                'exit': 0.3
            },
            'distracted': {
                'enter': 0.7,
                'exit': 0.4
            },
            'normal': {
                'enter': 0.5,
                'exit': 0.2
            }
        }
        
        # Minimum duration requirements (seconds)
        self.min_state_durations = {
            'sleeping': 2.0,    # Must be sleeping for at least 2 seconds
            'drowsy': 1.5,      # Must be drowsy for at least 1.5 seconds
            'distracted': 3.0,  # Must be distracted for at least 3 seconds
            'normal': 1.0       # Can return to normal quickly
        }
    
    def should_transition(self, current_state: str, proposed_state: str, 
                         confidence: float, current_duration: float) -> bool:
        """Determine if state transition should occur based on hysteresis"""
        
        # If staying in same state, no transition needed
        if current_state == proposed_state:
            return False
        
        # Check minimum duration requirement for current state
        if current_duration < self.min_state_durations.get(current_state, 0.0):
            return False
        
        # Check confidence threshold for entering new state
        enter_threshold = self.state_thresholds.get(proposed_state, {}).get('enter', 0.5)
        
        if confidence >= enter_threshold:
            return True
        
        # Special case: exiting current state with low confidence
        exit_threshold = self.state_thresholds.get(current_state, {}).get('exit', 0.3)
        if confidence < exit_threshold:
            return True
        
        return False

class AdvancedStateClassifier:
    """Advanced state classifier with hysteresis and transition management"""
    
    def __init__(self):
        self.config = SleepDetectionConfig()
        self.state_history = StateHistory()
        self.hysteresis_classifier = HysteresisClassifier(self.config)
        
        # Classification weights for different factors
        self.classification_weights = {
            'eye_closure': 0.4,
            'blink_pattern': 0.2,
            'head_movement': 0.3,
            'temporal_consistency': 0.1
        }
        
        logger.info("Advanced State Classifier initialized")
    
    def classify_state(self, metrics: SleepMetrics, raw_confidence: float) -> Tuple[str, float, str]:
        """
        Classify state with advanced logic
        Returns: (state, confidence, reason)
        """
        
        # Calculate individual classification scores
        sleep_score = self._calculate_sleep_score(metrics)
        drowsy_score = self._calculate_drowsy_score(metrics)
        distracted_score = self._calculate_distracted_score(metrics)
        normal_score = self._calculate_normal_score(metrics)
        
        # Determine proposed state and confidence
        scores = {
            'sleeping': sleep_score,
            'drowsy': drowsy_score,
            'distracted': distracted_score,
            'normal': normal_score
        }
        
        proposed_state = max(scores, key=scores.get)
        proposed_confidence = scores[proposed_state]
        
        # Generate reason for classification
        reason = self._generate_classification_reason(proposed_state, metrics)
        
        # Apply hysteresis logic
        current_duration = self.state_history.get_current_state_duration()
        
        if self.hysteresis_classifier.should_transition(
            self.state_history.current_state, 
            proposed_state, 
            proposed_confidence, 
            current_duration
        ):
            # Perform state transition
            self.state_history.add_transition(proposed_state, proposed_confidence, reason)
            return proposed_state, proposed_confidence, reason
        else:
            # Stay in current state
            return self.state_history.current_state, raw_confidence, f"Maintaining {self.state_history.current_state}"
    
    def _calculate_sleep_score(self, metrics: SleepMetrics) -> float:
        """Calculate sleep classification score"""
        score = 0.0
        
        # Primary indicator: eye closure duration
        if metrics.eye_closure_duration >= self.config.SLEEP_EYE_CLOSURE_THRESHOLD:
            score += 80.0
        elif metrics.eye_closure_duration >= 2.0:
            score += 60.0
        elif metrics.eye_closure_duration >= 1.0:
            score += 30.0
        
        # Secondary indicator: very low eye aspect ratio
        if metrics.eye_aspect_ratio < 0.15:
            score += 15.0
        elif metrics.eye_aspect_ratio < 0.2:
            score += 10.0
        
        # Tertiary indicator: very low blink rate (microsleep)
        if metrics.blink_rate < 3:
            score += 5.0
        
        return min(100.0, score)
    
    def _calculate_drowsy_score(self, metrics: SleepMetrics) -> float:
        """Calculate drowsiness classification score"""
        score = 0.0
        
        # Primary indicator: high blink rate
        if metrics.blink_rate > self.config.DROWSY_BLINK_RATE_THRESHOLD:
            score += 50.0
        elif metrics.blink_rate > 15:
            score += 30.0
        
        # Secondary indicator: brief eye closures
        if 0.5 <= metrics.eye_closure_duration < self.config.SLEEP_EYE_CLOSURE_THRESHOLD:
            score += 40.0
        
        # Tertiary indicator: low eye aspect ratio
        if 0.2 <= metrics.eye_aspect_ratio < 0.25:
            score += 20.0
        
        # Quaternary indicator: moderate drowsiness score
        if metrics.drowsiness_score > 40:
            score += 10.0
        
        return min(100.0, score)
    
    def _calculate_distracted_score(self, metrics: SleepMetrics) -> float:
        """Calculate distraction classification score"""
        score = 0.0
        
        # Primary indicator: head movement angle
        if metrics.head_movement_angle > self.config.DISTRACTION_HEAD_ANGLE_THRESHOLD:
            score += 60.0
        elif metrics.head_movement_angle > 10:
            score += 30.0
        
        # Secondary indicator: low head stability
        if metrics.head_stability < 0.5:
            score += 30.0
        elif metrics.head_stability < 0.7:
            score += 15.0
        
        # Tertiary indicator: high distraction score
        if metrics.distraction_score > 50:
            score += 20.0
        
        return min(100.0, score)
    
    def _calculate_normal_score(self, metrics: SleepMetrics) -> float:
        """Calculate normal state classification score"""
        score = 100.0  # Start with perfect normal score
        
        # Reduce based on negative indicators
        if metrics.eye_closure_duration > 1.0:
            score -= 50.0
        
        if metrics.blink_rate > 25 or metrics.blink_rate < 5:
            score -= 20.0
        
        if metrics.head_movement_angle > 10:
            score -= 30.0
        
        if metrics.head_stability < 0.7:
            score -= 20.0
        
        # Boost based on positive indicators
        if (0.25 <= metrics.eye_aspect_ratio <= 0.4 and 
            5 <= metrics.blink_rate <= 20 and 
            metrics.head_stability > 0.8):
            score += 10.0
        
        return max(0.0, score)
    
    def _generate_classification_reason(self, state: str, metrics: SleepMetrics) -> str:
        """Generate human-readable reason for classification"""
        
        if state == 'sleeping':
            if metrics.eye_closure_duration >= self.config.SLEEP_EYE_CLOSURE_THRESHOLD:
                return f"Eyes closed for {metrics.eye_closure_duration:.1f}s"
            else:
                return "Extended eye closure detected"
        
        elif state == 'drowsy':
            if metrics.blink_rate > self.config.DROWSY_BLINK_RATE_THRESHOLD:
                return f"High blink rate: {metrics.blink_rate:.0f}/min"
            elif metrics.eye_closure_duration > 0.5:
                return f"Brief eye closures: {metrics.eye_closure_duration:.1f}s"
            else:
                return "Drowsiness indicators detected"
        
        elif state == 'distracted':
            if metrics.head_movement_angle > self.config.DISTRACTION_HEAD_ANGLE_THRESHOLD:
                return f"Head turned {metrics.head_movement_angle:.1f}°"
            elif metrics.head_stability < 0.5:
                return f"Head instability: {metrics.head_stability:.2f}"
            else:
                return "Attention diverted"
        
        else:  # normal
            return "Alert and focused"
    
    def get_current_state(self) -> str:
        """Get current state"""
        return self.state_history.current_state
    
    def get_state_duration(self) -> float:
        """Get current state duration in seconds"""
        return self.state_history.get_current_state_duration()
    
    def get_state_summary(self) -> Dict[str, any]:
        """Get comprehensive state summary"""
        recent_transitions = self.state_history.get_recent_transitions(5)
        
        return {
            'current_state': self.state_history.current_state,
            'state_duration': self.get_state_duration(),
            'total_transitions': len(self.state_history.transitions),
            'recent_transitions': len(recent_transitions),
            'state_durations': self.state_history.state_durations.copy(),
            'last_transition': recent_transitions[-1] if recent_transitions else None
        }
    
    def reset_classifier(self):
        """Reset classifier state"""
        self.state_history = StateHistory()
        logger.info("State classifier reset")