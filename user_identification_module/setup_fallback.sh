#!/bin/bash
# Setup script for Face Recognition System with Fallback Support

echo "🚀 Setting up Face Recognition System with Fallback Support..."

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run this from the face_detection_module directory."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if MongoDB is installed
echo "🔍 Checking MongoDB installation..."
if command -v mongod &> /dev/null; then
    echo "✅ MongoDB is already installed"
else
    echo "❌ MongoDB not found. Installing MongoDB..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "🍺 Installing MongoDB using Homebrew..."
            brew tap mongodb/brew
            brew install mongodb-community
        else
            echo "❌ Homebrew not found. Please install MongoDB manually:"
            echo "   Visit: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/"
            exit 1
        fi
    else
        echo "❌ Please install MongoDB manually for your system:"
        echo "   Visit: https://docs.mongodb.com/manual/installation/"
        exit 1
    fi
fi

# Start MongoDB service
echo "🚀 Starting MongoDB service..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew services start mongodb/brew/mongodb-community
else
    # Linux (systemd)
    sudo systemctl start mongod
    sudo systemctl enable mongod
fi

# Wait a moment for MongoDB to start
echo "⏳ Waiting for MongoDB to start..."
sleep 3

# Test MongoDB connection
echo "🔍 Testing MongoDB connection..."
python3 -c "
try:
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.server_info()
    print('✅ MongoDB connection successful')
except Exception as e:
    print(f'❌ MongoDB connection failed: {e}')
    print('💡 Make sure MongoDB is running: brew services start mongodb/brew/mongodb-community')
"

# Test face_recognition library
echo "🔍 Testing face_recognition library..."
python3 -c "
try:
    import face_recognition
    print('✅ face_recognition library working')
except Exception as e:
    print(f'❌ face_recognition library error: {e}')
    print('💡 You may need to install cmake: brew install cmake')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your AWS credentials (optional)"
echo "2. Start the web UI: python web_ui.py"
echo "3. Open browser to: http://localhost:5000"
echo ""
echo "🔧 System will automatically:"
echo "   • Use AWS Rekognition if credentials are available"
echo "   • Fall back to OpenCV + MongoDB if AWS is not available"
echo "   • Use SQLite as final fallback"
echo ""
echo "📊 Check system status in the web UI for current mode"