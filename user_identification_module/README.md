# Face Recognition System with Intelligent Fallback

A robust Python-based face recognition system that automatically adapts to available services:
- **Primary**: AWS Rekognition + SQLite
- **Fallback**: OpenCV + MongoDB + face_recognition library
- **Emergency**: OpenCV + SQLite (limited recognition)

## Features

- **🔄 Intelligent Fallback System** - Automatically switches between AWS and local processing
- **🎯 Multiple Detection Methods** - AWS Rekognition, OpenCV, and face_recognition library
- **💾 Flexible Storage** - MongoDB for advanced features, SQLite for compatibility
- **📹 Real-time Processing** - Live camera feed with instant recognition
- **🌐 Web Interface** - Modern browser-based UI for easy testing
- **📊 System Monitoring** - Real-time status of all components
- **🔧 Auto-Configuration** - Detects available services and configures automatically

## Quick Setup

### Automated Setup (Recommended)

```bash
cd face_detection_module
./setup_fallback.sh
```

This script will:
- Install all Python dependencies
- Install and configure MongoDB
- Test all system components
- Provide setup status report

### Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install MongoDB** (for fallback mode)
   ```bash
   # macOS
   brew tap mongodb/brew
   brew install mongodb-community
   brew services start mongodb/brew/mongodb-community
   
   # Ubuntu/Debian
   sudo apt-get install mongodb
   sudo systemctl start mongodb
   ```

3. **AWS Configuration** (optional)
   - Copy `.env.example` to `.env`
   - Add your AWS credentials:
     ```
     AWS_ACCESS_KEY_ID=your_access_key_here
     AWS_SECRET_ACCESS_KEY=your_secret_key_here
     AWS_REGION=us-east-1
     ```

4. **AWS Permissions** (if using AWS)
   - `rekognition:CreateCollection`
   - `rekognition:DescribeCollection`
   - `rekognition:IndexFaces`
   - `rekognition:SearchFacesByImage`
   - `rekognition:DetectFaces`
   - `rekognition:DeleteFaces`

## Usage

### Option 1: Web UI (Recommended for Testing)

```bash
python web_ui.py
```

Then open your browser to: `http://localhost:5000`

**Web UI Features:**
- Live camera feed in browser
- Click-to-scan face recognition
- Easy user registration form
- Real-time status updates
- User list management

### Option 2: Command Line Interface

```bash
python main.py
```

**CLI Menu Options:**
1. **Start Camera & Recognition** - Opens camera feed with controls:
   - Press `s` to scan for existing users
   - Press `a` to add a new user
   - Press `q` to return to main menu

2. **List Registered Users** - Shows all users in the database

3. **Exit** - Quit the application

## Web UI Usage Guide

1. **Start the Web UI**: Run `python web_ui.py`
2. **Open Browser**: Go to `http://localhost:5000`
3. **Start Camera**: Click "📷 Start Camera" button
4. **Scan for Users**: Position your face and click "🔍 Scan for User"
5. **Add New Users**: 
   - Enter name and optional JSON data
   - Position your face in camera
   - Click "➕ Add New User"
6. **View Users**: Click "🔄 Refresh List" to see registered users

## System Architecture

### Core Components
- **`fallback_face_system.py`** - Intelligent system orchestrator with auto-fallback
- **`web_ui.py`** - Modern web interface with real-time status
- **`camera.py`** - Camera management and frame processing

### Face Recognition Services
- **`rekognition_service.py`** - AWS Rekognition integration
- **`opencv_face_service.py`** - Local OpenCV + face_recognition processing

### Database Services
- **`mongodb_service.py`** - MongoDB operations with face encodings
- **`database.py`** - SQLite operations for compatibility

### Utilities
- **`config.py`** - Unified configuration management
- **`debug_face_detection.py`** - Comprehensive debugging tools
- **`setup_fallback.sh`** - Automated setup script

## Database Schema

The system uses SQLite with the following schema:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Endpoints (Web UI)

- `POST /api/start_camera` - Start camera streaming
- `POST /api/stop_camera` - Stop camera streaming
- `POST /api/scan_face` - Scan current frame for faces
- `POST /api/add_user` - Add new user with current frame
- `GET /api/list_users` - List all registered users

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Try changing `CAMERA_INDEX` in config.py (usually 0 or 1)
- For web UI, allow camera permissions in your browser

### AWS Issues
- Verify your AWS credentials are correct
- Check that your AWS region supports Rekognition
- Ensure you have the required permissions

### Face Detection Issues
- Ensure good lighting conditions
- Face should be clearly visible and facing the camera
- Adjust `CONFIDENCE_THRESHOLD` in config.py if needed

### Web UI Issues
- Make sure port 5000 is not in use by another application
- Check browser console for JavaScript errors
- Ensure Flask and Flask-SocketIO are installed

## System Modes

### 🔄 Auto Mode (Default)
- Tries AWS Rekognition first
- Falls back to OpenCV + MongoDB if AWS unavailable
- Uses SQLite as final fallback

### ⚙️ Configuration Options
Set in `.env` file:
```bash
FALLBACK_MODE=auto          # auto, aws_only, fallback_only
FACE_RECOGNITION_TOLERANCE=0.6  # Lower = stricter matching
MONGODB_URI=mongodb://localhost:27017/
```

## Notes

- **Face encodings** stored in MongoDB for local recognition
- **No face images** stored locally - only mathematical encodings
- **Automatic service detection** - system adapts to available services
- **Real-time status monitoring** via web interface
- **Cross-platform compatibility** - works on macOS, Linux, Windows