#!/usr/bin/env python3
"""
Quick Start Script for Sleep Detection Module
Simplified entry point for immediate use
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Quick start the sleep detection module"""
    print("🚀 Sleep Detection Module - Quick Start")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if virtual environment exists
    venv_exists = os.path.exists('.venv')
    
    if not venv_exists:
        print("🐍 Setting up virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            print("💡 Falling back to system Python")
    
    # Determine Python executable
    if os.name == 'nt':  # Windows
        venv_python = '.venv\\Scripts\\python.exe'
        activation_cmd = '.venv\\Scripts\\activate.bat && '
    else:  # macOS/Linux
        venv_python = '.venv/bin/python'
        activation_cmd = 'source .venv/bin/activate && '
    
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    # Install requirements if needed
    if os.path.exists('requirements.txt'):
        print("📦 Installing requirements...")
        try:
            if venv_exists and os.path.exists(venv_python):
                subprocess.run([python_exe, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                             check=True, capture_output=True)
            else:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                             check=True, capture_output=True)
            print("✅ Requirements installed")
        except subprocess.CalledProcessError:
            print("⚠️  Could not install requirements automatically")
    
    # Setup .env if needed
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        import shutil
        shutil.copy('.env.example', '.env')
        print("📋 Created .env file from template")
        print("⚠️  Please edit .env with your AWS credentials")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    print("\n🎯 Choose how to run the system:")
    print("1. Web Interface (Recommended)")
    print("2. Command Line")
    print("3. Visual Demo")
    print("4. Full Setup Menu")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            print("\n🌐 Starting Web Interface...")
            print("📱 Open http://127.0.0.1:5001 in your browser")
            print("🛑 Press Ctrl+C to stop")
            subprocess.run([python_exe, 'sleep_web_ui.py'])
            
        elif choice == '2':
            print("\n💻 Starting Command Line System...")
            subprocess.run([python_exe, 'sleep_detection_system.py'])
            
        elif choice == '3':
            print("\n🎬 Starting Visual Demo...")
            subprocess.run([python_exe, 'visual_demo.py'])
            
        elif choice == '4':
            print("\n🔧 Opening Full Setup Menu...")
            subprocess.run([python_exe, 'setup_and_run.py'])
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try running: python setup_and_run.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == '__main__':
    main()