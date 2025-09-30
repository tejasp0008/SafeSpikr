"""
Visual Overlay Manager
Advanced visual overlays and real-time feedback for sleep detection
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import math
from sleep_detection_engine import SleepMetrics, DetectionResult
from sleep_config import SleepDetectionConfig

class OverlayColors:
    """Color constants for overlays"""
    # State colors
    NORMAL = (0, 255, 0)      # Green
    DROWSY = (0, 255, 255)    # Yellow
    SLEEPING = (0, 0, 255)    # Red
    DISTRACTED = (255, 0, 255) # Magenta
    UNKNOWN = (128, 128, 128)  # Gray
    
    # UI colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    
    # Transparency
    OVERLAY_ALPHA = 0.7

class FacialLandmarkVisualizer:
    """Visualize facial landmarks on video feed"""
    
    def __init__(self):
        self.landmark_colors = {
            'eyes': (0, 255, 255),      # Yellow
            'eyebrows': (255, 255, 0),  # Cyan
            'nose': (0, 255, 0),        # Green
            'mouth': (255, 0, 255),     # Magenta
            'face_outline': (255, 255, 255)  # White
        }
    
    def draw_eye_landmarks(self, frame: np.ndarray, eye_points: List[Tuple[float, float]], 
                          frame_width: int, frame_height: int, color: Tuple[int, int, int]):
        """Draw eye landmark points"""
        if not eye_points or len(eye_points) < 6:
            return
        
        # Convert normalized coordinates to pixel coordinates
        pixel_points = []
        for x, y in eye_points:
            px = int(x * frame_width)
            py = int(y * frame_height)
            pixel_points.append((px, py))
        
        # Draw eye outline
        if len(pixel_points) >= 6:
            eye_outline = np.array(pixel_points, np.int32)
            cv2.polylines(frame, [eye_outline], True, color, 2)
            
            # Draw individual points
            for point in pixel_points:
                cv2.circle(frame, point, 2, color, -1)
    
    def draw_facial_landmarks(self, frame: np.ndarray, landmarks: Any) -> np.ndarray:
        """Draw all facial landmarks"""
        if not landmarks:
            return frame
        
        height, width = frame.shape[:2]
        
        # Draw eyes
        if hasattr(landmarks, 'left_eye') and landmarks.left_eye:
            self.draw_eye_landmarks(frame, landmarks.left_eye, width, height, 
                                  self.landmark_colors['eyes'])
        
        if hasattr(landmarks, 'right_eye') and landmarks.right_eye:
            self.draw_eye_landmarks(frame, landmarks.right_eye, width, height, 
                                  self.landmark_colors['eyes'])
        
        # Draw nose
        if hasattr(landmarks, 'nose') and landmarks.nose:
            for x, y in landmarks.nose:
                px = int(x * width)
                py = int(y * height)
                cv2.circle(frame, (px, py), 3, self.landmark_colors['nose'], -1)
        
        # Draw mouth
        if hasattr(landmarks, 'mouth') and landmarks.mouth:
            mouth_points = []
            for x, y in landmarks.mouth:
                px = int(x * width)
                py = int(y * height)
                mouth_points.append((px, py))
                cv2.circle(frame, (px, py), 2, self.landmark_colors['mouth'], -1)
            
            # Draw mouth outline if we have enough points
            if len(mouth_points) >= 3:
                mouth_outline = np.array(mouth_points, np.int32)
                cv2.polylines(frame, [mouth_outline], True, self.landmark_colors['mouth'], 2)
        
        return frame

class MetricsVisualizer:
    """Visualize detection metrics and gauges"""
    
    def __init__(self):
        self.gauge_radius = 40
        self.gauge_thickness = 8
    
    def draw_circular_gauge(self, frame: np.ndarray, center: Tuple[int, int], 
                           value: float, max_value: float, color: Tuple[int, int, int],
                           label: str) -> np.ndarray:
        """Draw a circular gauge for metrics"""
        x, y = center
        
        # Background circle
        cv2.circle(frame, center, self.gauge_radius, OverlayColors.WHITE, 2)
        
        # Calculate angle for value (0-360 degrees)
        angle = int((value / max_value) * 360) if max_value > 0 else 0
        angle = min(360, max(0, angle))
        
        # Draw arc for current value
        if angle > 0:
            # Convert to OpenCV ellipse format (start_angle, end_angle)
            start_angle = -90  # Start from top
            end_angle = start_angle + angle
            
            cv2.ellipse(frame, center, (self.gauge_radius, self.gauge_radius), 
                       0, start_angle, end_angle, color, self.gauge_thickness)
        
        # Draw center dot
        cv2.circle(frame, center, 5, color, -1)
        
        # Draw value text
        value_text = f"{value:.1f}"
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(frame, value_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, OverlayColors.WHITE, 1)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        label_x = x - label_size[0] // 2
        label_y = y + self.gauge_radius + 20
        cv2.putText(frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, OverlayColors.WHITE, 1)
        
        return frame
    
    def draw_progress_bar(self, frame: np.ndarray, position: Tuple[int, int], 
                         width: int, height: int, value: float, max_value: float,
                         color: Tuple[int, int, int], label: str) -> np.ndarray:
        """Draw a horizontal progress bar"""
        x, y = position
        
        # Background rectangle
        cv2.rectangle(frame, (x, y), (x + width, y + height), OverlayColors.WHITE, 2)
        
        # Fill rectangle based on value
        fill_width = int((value / max_value) * width) if max_value > 0 else 0
        fill_width = min(width, max(0, fill_width))
        
        if fill_width > 0:
            cv2.rectangle(frame, (x + 2, y + 2), (x + fill_width - 2, y + height - 2), 
                         color, -1)
        
        # Draw value text
        value_text = f"{label}: {value:.1f}"
        cv2.putText(frame, value_text, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, OverlayColors.WHITE, 1)
        
        return frame

class AlertVisualizer:
    """Visualize alerts and warnings"""
    
    def __init__(self):
        self.alert_duration = 2.0  # seconds
        self.pulse_frequency = 2.0  # Hz
    
    def draw_state_alert(self, frame: np.ndarray, state: str, confidence: float,
                        timestamp: datetime) -> np.ndarray:
        """Draw state-based alert overlay"""
        height, width = frame.shape[:2]
        
        # Get state color
        color_map = {
            'normal': OverlayColors.NORMAL,
            'drowsy': OverlayColors.DROWSY,
            'sleeping': OverlayColors.SLEEPING,
            'distracted': OverlayColors.DISTRACTED
        }
        color = color_map.get(state, OverlayColors.UNKNOWN)
        
        # Create pulsing effect for non-normal states
        pulse_alpha = 1.0
        if state != 'normal':
            elapsed = (datetime.now() - timestamp).total_seconds()
            pulse_alpha = 0.5 + 0.5 * math.sin(elapsed * self.pulse_frequency * 2 * math.pi)
        
        # Draw border around frame
        border_thickness = 10 if state != 'normal' else 5
        border_color = tuple(int(c * pulse_alpha) for c in color)
        
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color, border_thickness)
        
        # Draw state indicator in corner
        indicator_size = 100
        indicator_pos = (width - indicator_size - 20, 20)
        
        # Background for indicator
        cv2.rectangle(frame, indicator_pos, 
                     (indicator_pos[0] + indicator_size, indicator_pos[1] + 60),
                     OverlayColors.BLACK, -1)
        cv2.rectangle(frame, indicator_pos, 
                     (indicator_pos[0] + indicator_size, indicator_pos[1] + 60),
                     color, 2)
        
        # State text
        state_text = state.upper()
        text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_BOLD, 0.7, 2)[0]
        text_x = indicator_pos[0] + (indicator_size - text_size[0]) // 2
        text_y = indicator_pos[1] + 25
        cv2.putText(frame, state_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_BOLD, 0.7, color, 2)
        
        # Confidence text
        conf_text = f"{confidence:.1f}%"
        conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        conf_x = indicator_pos[0] + (indicator_size - conf_size[0]) // 2
        conf_y = indicator_pos[1] + 50
        cv2.putText(frame, conf_text, (conf_x, conf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, OverlayColors.WHITE, 1)
        
        return frame
    
    def draw_warning_message(self, frame: np.ndarray, message: str, 
                           warning_type: str = 'warning') -> np.ndarray:
        """Draw warning message overlay"""
        height, width = frame.shape[:2]
        
        # Choose color based on warning type
        color_map = {
            'warning': OverlayColors.DROWSY,
            'alert': OverlayColors.SLEEPING,
            'info': OverlayColors.BLUE
        }
        color = color_map.get(warning_type, OverlayColors.DROWSY)
        
        # Calculate text size and position
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        
        # Background rectangle
        padding = 10
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     OverlayColors.BLACK, -1)
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     color, 2)
        
        # Warning text
        cv2.putText(frame, message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame

class VisualOverlayManager:
    """Main visual overlay manager"""
    
    def __init__(self):
        self.config = SleepDetectionConfig()
        self.landmark_visualizer = FacialLandmarkVisualizer()
        self.metrics_visualizer = MetricsVisualizer()
        self.alert_visualizer = AlertVisualizer()
        
        # Overlay settings
        self.show_landmarks = True
        self.show_metrics = True
        self.show_alerts = True
        self.show_fps = True
        self.show_timestamp = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = datetime.now()
        self.current_fps = 0.0
    
    def create_comprehensive_overlay(self, frame: np.ndarray, 
                                   detection_result: DetectionResult,
                                   landmarks: Any = None,
                                   system_info: Dict[str, Any] = None) -> np.ndarray:
        """Create comprehensive visual overlay"""
        if frame is None:
            return frame
        
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Update FPS counter
        self._update_fps_counter()
        
        # 1. Draw facial landmarks
        if self.show_landmarks and landmarks:
            overlay_frame = self.landmark_visualizer.draw_facial_landmarks(
                overlay_frame, landmarks
            )
        
        # 2. Draw state alert
        if self.show_alerts and detection_result:
            overlay_frame = self.alert_visualizer.draw_state_alert(
                overlay_frame, detection_result.state, detection_result.confidence,
                detection_result.timestamp
            )
        
        # 3. Draw metrics gauges
        if self.show_metrics and detection_result and detection_result.metrics:
            overlay_frame = self._draw_metrics_panel(overlay_frame, detection_result.metrics)
        
        # 4. Draw system information
        if system_info:
            overlay_frame = self._draw_system_info(overlay_frame, system_info)
        
        # 5. Draw FPS counter
        if self.show_fps:
            overlay_frame = self._draw_fps_counter(overlay_frame)
        
        # 6. Draw timestamp
        if self.show_timestamp:
            overlay_frame = self._draw_timestamp(overlay_frame)
        
        # 7. Draw warnings if needed
        overlay_frame = self._draw_contextual_warnings(overlay_frame, detection_result)
        
        return overlay_frame
    
    def _draw_metrics_panel(self, frame: np.ndarray, metrics: SleepMetrics) -> np.ndarray:
        """Draw metrics panel with gauges and bars"""
        height, width = frame.shape[:2]
        
        # Panel background
        panel_x = 20
        panel_y = 20
        panel_width = 200
        panel_height = 300
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     OverlayColors.BLACK, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     OverlayColors.WHITE, 2)
        
        # Title
        cv2.putText(frame, "METRICS", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_BOLD, 0.6, OverlayColors.WHITE, 2)
        
        # Eye closure gauge
        gauge_y = panel_y + 60
        self.metrics_visualizer.draw_circular_gauge(
            frame, (panel_x + 50, gauge_y), 
            metrics.eye_closure_duration, 5.0, OverlayColors.SLEEPING, "Eye Closure"
        )
        
        # Blink rate gauge
        self.metrics_visualizer.draw_circular_gauge(
            frame, (panel_x + 150, gauge_y), 
            metrics.blink_rate, 30.0, OverlayColors.DROWSY, "Blink Rate"
        )
        
        # Head angle progress bar
        bar_y = gauge_y + 80
        self.metrics_visualizer.draw_progress_bar(
            frame, (panel_x + 10, bar_y), 180, 20,
            metrics.head_movement_angle, 45.0, OverlayColors.DISTRACTED, "Head Angle"
        )
        
        # Attention score progress bar
        bar_y += 40
        self.metrics_visualizer.draw_progress_bar(
            frame, (panel_x + 10, bar_y), 180, 20,
            metrics.attention_score, 100.0, OverlayColors.NORMAL, "Attention"
        )
        
        # Drowsiness score progress bar
        bar_y += 40
        self.metrics_visualizer.draw_progress_bar(
            frame, (panel_x + 10, bar_y), 180, 20,
            metrics.drowsiness_score, 100.0, OverlayColors.DROWSY, "Drowsiness"
        )
        
        # Eye aspect ratio
        bar_y += 40
        self.metrics_visualizer.draw_progress_bar(
            frame, (panel_x + 10, bar_y), 180, 20,
            metrics.eye_aspect_ratio * 100, 50.0, OverlayColors.CYAN, "Eye Opening"
        )
        
        return frame
    
    def _draw_system_info(self, frame: np.ndarray, system_info: Dict[str, Any]) -> np.ndarray:
        """Draw system information panel"""
        height, width = frame.shape[:2]
        
        # Position in top right
        info_x = width - 250
        info_y = 20
        info_width = 230
        info_height = 120
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), 
                     (info_x + info_width, info_y + info_height),
                     OverlayColors.BLACK, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Border
        cv2.rectangle(frame, (info_x, info_y), 
                     (info_x + info_width, info_y + info_height),
                     OverlayColors.WHITE, 2)
        
        # Title
        cv2.putText(frame, "SYSTEM INFO", (info_x + 10, info_y + 25), 
                   cv2.FONT_HERSHEY_BOLD, 0.5, OverlayColors.WHITE, 1)
        
        # System information
        y_offset = info_y + 45
        line_height = 15
        
        # Detection method
        method = system_info.get('detection_method', 'Unknown')
        cv2.putText(frame, f"Method: {method}", (info_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, OverlayColors.WHITE, 1)
        y_offset += line_height
        
        # Processing FPS
        fps = system_info.get('processing_fps', 0)
        cv2.putText(frame, f"Processing: {fps:.1f} FPS", (info_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, OverlayColors.WHITE, 1)
        y_offset += line_height
        
        # Detection accuracy
        accuracy = system_info.get('detection_accuracy', 0)
        cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (info_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, OverlayColors.WHITE, 1)
        y_offset += line_height
        
        # Service status indicators
        aws_status = "●" if system_info.get('aws_available', False) else "○"
        opencv_status = "●" if system_info.get('opencv_available', False) else "○"
        camera_status = "●" if system_info.get('camera_available', False) else "○"
        
        cv2.putText(frame, f"AWS {aws_status} OpenCV {opencv_status} Cam {camera_status}", 
                   (info_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, OverlayColors.WHITE, 1)
        
        return frame
    
    def _draw_fps_counter(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter"""
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, OverlayColors.WHITE, 2)
        return frame
    
    def _draw_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """Draw current timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x = frame.shape[1] - text_size[0] - 10
        y = frame.shape[0] - 10
        
        cv2.putText(frame, timestamp, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, OverlayColors.WHITE, 2)
        return frame
    
    def _draw_contextual_warnings(self, frame: np.ndarray, 
                                 detection_result: DetectionResult) -> np.ndarray:
        """Draw contextual warnings based on detection state"""
        if not detection_result:
            return frame
        
        state = detection_result.state
        confidence = detection_result.confidence
        
        # Generate appropriate warnings
        if state == 'sleeping' and confidence > 80:
            frame = self.alert_visualizer.draw_warning_message(
                frame, "⚠️ SLEEP DETECTED - WAKE UP!", 'alert'
            )
        elif state == 'drowsy' and confidence > 70:
            frame = self.alert_visualizer.draw_warning_message(
                frame, "😪 DROWSINESS DETECTED", 'warning'
            )
        elif state == 'distracted' and confidence > 75:
            frame = self.alert_visualizer.draw_warning_message(
                frame, "👀 ATTENTION REQUIRED", 'warning'
            )
        
        return frame
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        
        current_time = datetime.now()
        elapsed = (current_time - self.fps_start_time).total_seconds()
        
        if elapsed >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def toggle_landmarks(self):
        """Toggle landmark visualization"""
        self.show_landmarks = not self.show_landmarks
    
    def toggle_metrics(self):
        """Toggle metrics visualization"""
        self.show_metrics = not self.show_metrics
    
    def toggle_alerts(self):
        """Toggle alert visualization"""
        self.show_alerts = not self.show_alerts
    
    def set_overlay_options(self, landmarks: bool = None, metrics: bool = None, 
                           alerts: bool = None, fps: bool = None, timestamp: bool = None):
        """Set overlay display options"""
        if landmarks is not None:
            self.show_landmarks = landmarks
        if metrics is not None:
            self.show_metrics = metrics
        if alerts is not None:
            self.show_alerts = alerts
        if fps is not None:
            self.show_fps = fps
        if timestamp is not None:
            self.show_timestamp = timestamp
    
    def create_demo_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create demo overlay when no detection is active"""
        if frame is None:
            return frame
        
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Demo message
        demo_text = "Sleep Detection System - Demo Mode"
        text_size = cv2.getTextSize(demo_text, cv2.FONT_HERSHEY_BOLD, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        # Background for text
        cv2.rectangle(overlay_frame, 
                     (text_x - 20, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 20, text_y + 10),
                     OverlayColors.BLACK, -1)
        
        # Demo text
        cv2.putText(overlay_frame, demo_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_BOLD, 1.0, OverlayColors.WHITE, 2)
        
        # Instructions
        instructions = [
            "Start monitoring to see real-time detection",
            "Look at camera for normal state",
            "Close eyes for 3+ seconds to test sleep detection",
            "Turn head away to test distraction detection"
        ]
        
        y_offset = text_y + 50
        for instruction in instructions:
            inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            inst_x = (width - inst_size[0]) // 2
            cv2.putText(overlay_frame, instruction, (inst_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, OverlayColors.CYAN, 1)
            y_offset += 30
        
        return overlay_frame