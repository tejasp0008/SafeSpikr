#!/usr/bin/env python3
"""
Virtual Environment Setup for Face Detection Module
Creates and manages virtual environment for the face detection module
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def get_venv_python():
    """Get the path to the virtual environment Python executable"""
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "python.exe")
    else:
        return os.path.join(".venv", "bin", "python")

def get_venv_pip():
    """Get the path to the virtual environment pip executable"""
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "pip.exe")
    else:
        return os.path.join(".venv", "bin", "pip")

def setup_virtual_environment():
    """Setup virtual environment for face detection module"""
    print("🐍 Setting up Face Detection Module Virtual Environment...")
    
    venv_path = ".venv"
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists(venv_path):
        print("📦 Creating virtual environment...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'venv', venv_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Virtual environment created")
            else:
                print(f"❌ Failed to create virtual environment: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Upgrade pip in virtual environment
    venv_pip = get_venv_pip()
    if os.path.exists(venv_pip):
        print("📦 Upgrading pip in virtual environment...")
        try:
            subprocess.run([venv_pip, 'install', '--upgrade', 'pip'], 
                         capture_output=True, text=True, check=True)
            print("✅ Pip upgraded")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not upgrade pip: {e}")
    
    return True

def install_requirements():
    """Install requirements in virtual environment"""
    print("\n📦 Installing requirements...")
    
    if not os.path.exists('requirements.txt'):
        print("⚠️  No requirements.txt found")
        return True
    
    venv_pip = get_venv_pip()
    
    if not os.path.exists(venv_pip):
        print("❌ Virtual environment pip not found")
        return False
    
    try:
        result = subprocess.run([
            venv_pip, 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def show_activation_instructions():
    """Show activation instructions"""
    print("\n🎯 Virtual Environment Setup Complete!")
    print("=" * 50)
    
    system = platform.system()
    
    print("To activate the virtual environment:")
    
    if system == "Windows":
        print("\nFor Windows Command Prompt:")
        print("  .venv\\Scripts\\activate.bat")
        print("\nFor Windows PowerShell:")
        print("  .venv\\Scripts\\Activate.ps1")
        print("\nOr run the batch file:")
        print("  activate_venv.bat")
    else:
        print("\nFor macOS/Linux:")
        print("  source .venv/bin/activate")
        print("\nOr run the shell script:")
        print("  ./activate_venv.sh")
    
    print("\nAfter activation, you can run:")
    print("  python face_recognition_system.py")
    print("  python aws_face_system.py")
    print("  python web_ui.py")
    
    print("\nTo deactivate:")
    print("  deactivate")

def main():
    """Main setup function"""
    print("🚀 Face Detection Module Virtual Environment Setup")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to setup virtual environment")
        return False
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Show activation instructions
    show_activation_instructions()
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)