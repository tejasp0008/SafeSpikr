#!/usr/bin/env python3
"""
AWS-only Web UI for Face Recognition System
"""

import os
import base64
import json
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import threading
import time
from camera import CameraManager
from aws_face_system import AWSFaceSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aws_face_recognition_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class AWSWebFaceRecognitionSystem:
    def __init__(self):
        self.camera = CameraManager()
        self.face_system = AWSFaceSystem()
        self.streaming = False
        self.stream_thread = None
    
    def start_camera_stream(self):
        """Start streaming camera feed to web interface"""
        if not self.camera.start_camera():
            return False
        
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_frames)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        return True
    
    def stop_camera_stream(self):
        """Stop camera streaming"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1)
        self.camera.release_camera()
    
    def _stream_frames(self):
        """Stream camera frames to web interface"""
        while self.streaming:
            frame = self.camera.capture_frame()
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Emit frame to all connected clients
                socketio.emit('video_frame', {'image': frame_data})
            
            time.sleep(0.1)  # ~10 FPS
    
    def scan_current_frame(self):
        """Scan current camera frame for faces"""
        frame = self.camera.capture_frame()
        if frame is None:
            return {'success': False, 'message': 'No camera frame available'}
        
        # Convert frame to bytes
        image_bytes = self.camera.frame_to_bytes(frame)
        
        # Use AWS system to scan for user
        return self.face_system.scan_for_user(image_bytes)
    
    def add_user_from_frame(self, name, user_data):
        """Add a new user using current camera frame"""
        frame = self.camera.capture_frame()
        if frame is None:
            return {'success': False, 'message': 'No camera frame available'}
        
        # Convert frame to bytes
        image_bytes = self.camera.frame_to_bytes(frame)
        
        # Use AWS system to add user
        return self.face_system.add_user_complete(image_bytes, name, user_data)

# Global system instance
face_system = AWSWebFaceRecognitionSystem()

@app.route('/')
def index():
    """Main page"""
    return render_template('aws_index.html')

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera streaming"""
    if face_system.start_camera_stream():
        return jsonify({'success': True, 'message': 'Camera started'})
    else:
        return jsonify({'success': False, 'message': 'Failed to start camera'})

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera streaming"""
    face_system.stop_camera_stream()
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/api/scan_face', methods=['POST'])
def scan_face():
    """Scan for faces in current frame"""
    result = face_system.scan_current_frame()
    return jsonify(result)

@app.route('/api/add_user', methods=['POST'])
def add_user():
    """Add a new user"""
    data = request.get_json()
    name = data.get('name', '').strip()
    user_data = data.get('data', {})
    
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'})
    
    result = face_system.add_user_from_frame(name, user_data)
    return jsonify(result)

@app.route('/api/list_users', methods=['GET'])
def list_users():
    """List all registered users"""
    users = face_system.face_system.list_users()
    return jsonify({
        'success': True,
        'users': [{'face_id': face_id, 'name': name} for face_id, name in users]
    })

@app.route('/api/test_detection', methods=['POST'])
def test_detection():
    """Test AWS face detection"""
    frame = face_system.camera.capture_frame()
    if frame is None:
        return jsonify({'success': False, 'message': 'No camera frame available'})
    
    # Convert frame to bytes
    image_bytes = face_system.camera.frame_to_bytes(frame)
    
    # Test AWS detection
    try:
        faces = face_system.face_system.detect_faces(image_bytes)
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'message': f'AWS Rekognition detected {len(faces)} face(s)',
            'mode': 'aws'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'AWS detection failed: {str(e)}'
        })

@app.route('/api/system_status', methods=['GET'])
def system_status():
    """Get system status"""
    status = face_system.face_system.get_system_status()
    return jsonify({
        'success': True,
        'status': status
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting AWS Face Recognition Web UI...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("⚠️ Make sure your AWS credentials are configured in .env file")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    finally:
        face_system.stop_camera_stream()