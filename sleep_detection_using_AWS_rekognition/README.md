# Sleep Detection Module

A real-time sleep and distraction detection system using AWS Rekognition and OpenCV.

## Features

- **Real-time Detection**: Monitor sleep, distraction, and normal alertness states
- **AWS Rekognition Integration**: Cloud-based facial analysis for accurate detection
- **OpenCV Fallback**: Local processing when AWS is unavailable
- **Web Interface**: Live monitoring dashboard with visual feedback
- **Configurable Thresholds**: Customizable detection parameters
- **Multi-state Classification**: Sleep, distracted, and normal states with confidence scores

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and preferences
   ```

3. **Run the System**
   ```bash
   python sleep_detection_system.py
   ```

4. **Access Web Interface**
   Open http://127.0.0.1:5001 in your browser

## Configuration

### AWS Setup
- Set your AWS credentials in `.env`
- Ensure your AWS account has Rekognition permissions
- The system will automatically fall back to OpenCV if AWS is unavailable

### Detection Thresholds
- `SLEEP_EYE_CLOSURE_THRESHOLD`: Seconds of eye closure to trigger sleep detection (default: 3.0)
- `DROWSY_BLINK_RATE_THRESHOLD`: Blinks per minute to indicate drowsiness (default: 20.0)
- `DISTRACTION_HEAD_ANGLE_THRESHOLD`: Head angle in degrees to trigger distraction (default: 15.0)
- `DISTRACTION_DURATION_THRESHOLD`: Seconds of distraction to trigger alert (default: 5.0)

## Architecture

```
Camera Feed → Frame Processor → AWS Rekognition/OpenCV → Sleep Detection Engine → Web UI
```

## Detection States

- **Normal**: Alert and focused state
- **Drowsy**: Frequent blinking or brief eye closures
- **Sleeping**: Extended eye closure (3+ seconds)
- **Distracted**: Head movement away from forward position (5+ seconds)

## API Endpoints

- `GET /`: Main monitoring dashboard
- `GET /video_feed`: Live video stream with overlays
- `GET /api/status`: Current detection status
- `POST /api/config`: Update configuration thresholds

## Requirements

- Python 3.8+
- Webcam or camera device
- AWS account with Rekognition access (optional, fallback available)
- Modern web browser for UI

## License

MIT License