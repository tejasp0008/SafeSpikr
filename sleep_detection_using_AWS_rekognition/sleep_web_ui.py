#!/usr/bin/env python3
"""
Sleep Detection Web UI
Flask-based web interface for sleep detection system
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import json
import logging
from datetime import datetime
import threading
import time

from sleep_detection_system import SleepDetectionSystem
from sleep_config import SleepDetectionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global system instance
detection_system = None
config = SleepDetectionConfig()

# Video streaming
video_frame = None
video_lock = threading.Lock()

def initialize_system():
    """Initialize the sleep detection system"""
    global detection_system
    
    try:
        detection_system = SleepDetectionSystem()
        
        # Add callbacks for real-time updates
        detection_system.add_detection_callback(on_detection_update)
        detection_system.add_state_change_callback(on_state_change)
        
        logger.info("Sleep detection system initialized for web UI")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize sleep detection system: {e}")
        return False

def on_detection_update(detection_result):
    """Callback for detection updates"""
    global video_frame
    
    try:
        # Get current frame with overlays
        current_frame = detection_system.camera_manager.get_current_frame()
        if current_frame is not None:
            # Prepare overlay data
            overlay_data = {
                'state': detection_result.state,
                'confidence': detection_result.confidence,
                'metrics': detection_result.metrics
            }
            
            # Get landmarks if available
            landmarks = None
            if hasattr(detection_result, 'landmarks'):
                landmarks = detection_result.landmarks
            
            # Get system info
            system_status = detection_system.get_system_status()
            system_info = {
                'detection_method': system_status['current_state']['detection_method'],
                'processing_fps': system_status['performance']['frames_per_second'],
                'detection_accuracy': system_status['performance']['detection_accuracy'],
                'aws_available': system_status['services']['aws_available'],
                'opencv_available': system_status['services']['opencv_available'],
                'camera_available': system_status['services']['camera_available']
            }
            
            # Add comprehensive overlays
            overlayed_frame = detection_system.camera_manager.add_detection_overlay(
                current_frame, overlay_data, landmarks, system_info
            )
            
            # Update video frame for streaming
            with video_lock:
                video_frame = overlayed_frame
        else:
            # If no current frame, try to get a fresh frame from camera
            fresh_frame = detection_system.camera_manager.capture_frame()
            if fresh_frame is not None:
                # Add basic overlay showing it's live
                cv2.putText(fresh_frame, "Live Camera - Start Monitoring", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                with video_lock:
                    video_frame = fresh_frame
                
    except Exception as e:
        logger.error(f"Error in detection update callback: {e}")

def on_state_change(from_state, to_state, confidence):
    """Callback for state changes"""
    logger.info(f"State change: {from_state} -> {to_state} (confidence: {confidence:.1f}%)")

def generate_video_frames():
    """Generate video frames for streaming"""
    global video_frame
    
    while True:
        try:
            frame = None
            
            with video_lock:
                if video_frame is not None:
                    frame = video_frame.copy()
            
            # If no processed frame, try to get live camera frame
            if frame is None and detection_system and detection_system.camera_manager:
                try:
                    # Try to get current frame first
                    live_frame = detection_system.camera_manager.get_current_frame()
                    
                    # If no current frame, capture a fresh one
                    if live_frame is None:
                        live_frame = detection_system.camera_manager.capture_frame()
                    
                    if live_frame is not None:
                        frame = live_frame.copy()
                        
                        # Add status overlay
                        if detection_system.is_monitoring:
                            cv2.putText(frame, "Monitoring Active", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, "AWS Rekognition Detection", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        else:
                            cv2.putText(frame, "Live Camera Feed", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, "Click 'Start Monitoring' to begin detection", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        # Add timestamp
                        timestamp = time.strftime("%H:%M:%S")
                        cv2.putText(frame, timestamp, (frame.shape[1] - 100, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                except Exception as camera_error:
                    logger.debug(f"Camera access error: {camera_error}")
            
            # If still no frame, create placeholder
            if frame is None:
                frame = create_placeholder_frame()
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Error generating video frame: {e}")
            time.sleep(0.1)

def create_placeholder_frame():
    """Create a placeholder frame when no video is available"""
    import numpy as np
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add gradient background
    for y in range(frame.shape[0]):
        intensity = int(30 + (y / frame.shape[0]) * 50)
        frame[y, :] = [intensity // 3, intensity // 2, intensity]
    
    # Add text
    cv2.putText(frame, "Sleep Detection System", (160, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Camera Initializing...", (200, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "Please wait or check camera connection", (140, 260), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, "AWS Rekognition Ready", (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, timestamp, (frame.shape[1] - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_video_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    try:
        if detection_system is None:
            return jsonify({
                'error': 'System not initialized',
                'system': {'monitoring_active': False},
                'services': {'aws_available': False, 'opencv_available': False, 'camera_available': False},
                'current_state': {'state': 'unknown', 'confidence': 0.0, 'duration': 0.0, 'detection_method': 'none'},
                'performance': {'frames_per_second': 0.0, 'detection_accuracy': 0.0}
            })
        
        status = detection_system.get_system_status()
        
        # Add current metrics to the response
        current_state = detection_system.get_current_state()
        
        # Format metrics for JSON serialization
        metrics_dict = {}
        if current_state.metrics:
            metrics_dict = {
                'eye_closure_duration': current_state.metrics.eye_closure_duration,
                'blink_rate': current_state.metrics.blink_rate,
                'head_movement_angle': current_state.metrics.head_movement_angle,
                'drowsiness_score': current_state.metrics.drowsiness_score,
                'distraction_score': current_state.metrics.distraction_score,
                'attention_score': current_state.metrics.attention_score,
                'eye_aspect_ratio': current_state.metrics.eye_aspect_ratio,
                'head_stability': current_state.metrics.head_stability
            }
        
        # Update current state with metrics
        status['current_state']['metrics'] = metrics_dict
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start sleep detection monitoring"""
    try:
        if detection_system is None:
            return jsonify({'success': False, 'message': 'System not initialized'})
        
        if detection_system.start_monitoring():
            return jsonify({'success': True, 'message': 'Monitoring started successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to start monitoring'})
            
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop sleep detection monitoring"""
    try:
        if detection_system is None:
            return jsonify({'success': False, 'message': 'System not initialized'})
        
        detection_system.stop_monitoring()
        return jsonify({'success': True, 'message': 'Monitoring stopped successfully'})
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset the detection system"""
    try:
        if detection_system is None:
            return jsonify({'success': False, 'message': 'System not initialized'})
        
        detection_system.reset_system()
        return jsonify({'success': True, 'message': 'System reset successfully'})
        
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration"""
    try:
        if request.method == 'GET':
            # Return current configuration
            thresholds = config.get_detection_thresholds()
            return jsonify({
                'success': True,
                'thresholds': thresholds,
                'aws_configured': bool(config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY)
            })
        
        elif request.method == 'POST':
            # Update configuration
            if detection_system is None:
                return jsonify({'success': False, 'message': 'System not initialized'})
            
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'No configuration data provided'})
            
            # Update thresholds
            success = detection_system.configure_thresholds(data)
            
            if success:
                return jsonify({'success': True, 'message': 'Configuration updated successfully'})
            else:
                return jsonify({'success': False, 'message': 'Failed to update configuration'})
                
    except Exception as e:
        logger.error(f"Error managing config: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/summary')
def get_detection_summary():
    """Get detection summary and statistics"""
    try:
        if detection_system is None:
            return jsonify({'error': 'System not initialized'})
        
        summary = detection_system.get_detection_summary()
        state_summary = detection_system.state_classifier.get_state_summary()
        
        return jsonify({
            'detection_summary': summary,
            'state_summary': state_summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/info')
def get_camera_info():
    """Get camera information"""
    try:
        if detection_system is None:
            return jsonify({'error': 'System not initialized'})
        
        camera_info = detection_system.camera_manager.get_camera_info()
        camera_stats = detection_system.camera_manager.get_statistics()
        
        return jsonify({
            'camera_info': camera_info,
            'camera_stats': camera_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting camera info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        if detection_system is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'System not initialized',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        system_status = detection_system.get_system_status()
        
        # Determine overall health
        healthy = (
            system_status['services']['camera_available'] and
            (system_status['services']['aws_available'] or system_status['services']['opencv_available'])
        )
        
        return jsonify({
            'status': 'healthy' if healthy else 'degraded',
            'services': system_status['services'],
            'monitoring_active': system_status['system']['monitoring_active'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run the web UI"""
    print("🚀 Starting Sleep Detection Web UI...")
    
    # Initialize the detection system
    if not initialize_system():
        print("❌ Failed to initialize sleep detection system")
        return
    
    print("✅ Sleep detection system initialized")
    print(f"🌐 Starting web server on {config.WEB_HOST}:{config.WEB_PORT}")
    
    try:
        # Run Flask app
        app.run(
            host=config.WEB_HOST,
            port=config.WEB_PORT,
            debug=config.DEBUG_MODE,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web UI...")
    except Exception as e:
        print(f"❌ Error running web server: {e}")
    finally:
        # Cleanup
        if detection_system:
            detection_system.shutdown()
        print("👋 Web UI stopped")

if __name__ == '__main__':
    main()