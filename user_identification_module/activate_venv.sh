#!/bin/bash
# Activation script for face detection module virtual environment

echo "🐍 Activating Face Detection Module Virtual Environment..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating it now..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source .venv/bin/activate

echo "✅ Virtual environment activated"
echo "📦 Installing/updating requirements..."

# Install requirements if they exist
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Requirements installed"
else
    echo "⚠️  No requirements.txt found"
fi

echo ""
echo "🎯 Face Detection Module Environment Ready!"
echo "To deactivate, run: deactivate"
echo ""