#!/bin/bash
# Activation script for the face detection module virtual environment

echo "🚀 Activating Face Detection Module Virtual Environment..."
source .venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Installed packages:"
pip list --format=columns

echo ""
echo "🎯 Quick Start Commands:"
echo "  python web_ui.py     - Start web interface (recommended)"
echo "  python main.py       - Start CLI interface"
echo ""
echo "🌐 Web UI will be available at: http://localhost:5000"
echo ""
echo "💡 Don't forget to set up your .env file with AWS credentials!"