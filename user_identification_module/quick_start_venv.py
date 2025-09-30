#!/usr/bin/env python3
"""
Quick Start with Virtual Environment for Face Detection Module
Easy activation and running of face detection system
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def get_venv_python():
    """Get the path to the virtual environment Python executable"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\python.exe"
    else:
        return ".venv/bin/python"

def main():
    """Quick start with virtual environment"""
    print("🚀 Face Detection Module - Quick Start with Virtual Environment")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if virtual environment exists
    venv_python = get_venv_python()
    
    if not os.path.exists(venv_python):
        print("🐍 Virtual environment not found. Setting it up...")
        
        # Run setup script
        try:
            result = subprocess.run([sys.executable, 'setup_venv.py'], check=True)
            print("✅ Virtual environment setup completed")
        except subprocess.CalledProcessError:
            print("❌ Failed to setup virtual environment")
            print("💡 Try running: python setup_venv.py")
            return
        except FileNotFoundError:
            print("❌ setup_venv.py not found")
            return
    else:
        print("✅ Virtual environment found")
    
    # Show menu
    print("\n🎯 Choose how to run the face detection system:")
    print("1. AWS Face Recognition System")
    print("2. Fallback Face Recognition System") 
    print("3. Web Interface")
    print("4. Debug Face Detection")
    print("5. Show activation instructions")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            print("\n☁️ Starting AWS Face Recognition System...")
            subprocess.run([venv_python, 'aws_face_system.py'])
            
        elif choice == '2':
            print("\n🔄 Starting Fallback Face Recognition System...")
            subprocess.run([venv_python, 'fallback_face_system.py'])
            
        elif choice == '3':
            print("\n🌐 Starting Web Interface...")
            print("📱 Open http://127.0.0.1:5000 in your browser")
            subprocess.run([venv_python, 'web_ui.py'])
            
        elif choice == '4':
            print("\n🔍 Starting Debug Face Detection...")
            subprocess.run([venv_python, 'debug_face_detection.py'])
            
        elif choice == '5':
            show_activation_instructions()
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure the script exists in the face_detection_module directory")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def show_activation_instructions():
    """Show manual activation instructions"""
    print("\n🐍 Manual Virtual Environment Activation")
    print("=" * 40)
    
    system = platform.system()
    
    if system == "Windows":
        print("For Windows Command Prompt:")
        print("  .venv\\Scripts\\activate.bat")
        print("\nFor Windows PowerShell:")
        print("  .venv\\Scripts\\Activate.ps1")
        print("\nOr run the batch file:")
        print("  activate_venv.bat")
    else:
        print("For macOS/Linux:")
        print("  source .venv/bin/activate")
        print("\nOr run the shell script:")
        print("  ./activate_venv.sh")
    
    print("\nAfter activation, you can run:")
    print("  python aws_face_system.py")
    print("  python fallback_face_system.py")
    print("  python web_ui.py")
    print("  python debug_face_detection.py")

if __name__ == '__main__':
    main()