#!/bin/bash
# Simple startup script for Face Recognition System

echo "🚀 Starting Face Recognition System..."

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup first."
    exit 1
fi

# Check system status
echo "🔍 Checking system status..."
python -c "
from fallback_face_system import FallbackFaceSystem
system = FallbackFaceSystem()
status = system.get_system_status()
print(f'📊 System Mode: {status[\"mode\"].upper()}')
print(f'🔧 AWS Available: {\"✅\" if status[\"aws_available\"] else \"❌\"}')
print(f'🍃 MongoDB Available: {\"✅\" if status[\"mongodb_available\"] else \"❌\"}')
print(f'👁️  OpenCV Available: {\"✅\" if status[\"opencv_available\"] else \"❌\"}')
print(f'💾 SQLite Available: {\"✅\" if status[\"sqlite_available\"] else \"❌\"}')
"

echo ""
echo "🌐 Starting Web UI on http://localhost:5000"
echo "📱 Open your browser and navigate to the URL above"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Start the web UI
python web_ui.py