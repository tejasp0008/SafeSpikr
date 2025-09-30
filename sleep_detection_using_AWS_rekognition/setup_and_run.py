#!/usr/bin/env python3
"""
Setup and Run Script for Sleep Detection Module
Easy setup and execution of the sleep detection system with virtual environment
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

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
    """Setup virtual environment"""
    print("\n🐍 Setting up virtual environment...")
    
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

def is_in_venv():
    """Check if we're running in the virtual environment"""
    venv_python = get_venv_python()
    return os.path.exists(venv_python) and os.path.samefile(sys.executable, venv_python)

def install_requirements():
    """Install required packages in virtual environment"""
    print("\n📦 Installing required packages...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists('requirements.txt'):
            print("❌ requirements.txt not found")
            return False
        
        # Use virtual environment pip if available
        pip_executable = get_venv_pip() if os.path.exists(get_venv_pip()) else sys.executable
        
        if pip_executable == sys.executable:
            # Use current Python's pip
            install_cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
        else:
            # Use virtual environment pip directly
            install_cmd = [pip_executable, 'install', '-r', 'requirements.txt']
        
        print(f"Installing with: {' '.join(install_cmd)}")
        
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_environment():
    """Setup environment configuration"""
    print("\n🔧 Setting up environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("📋 Creating .env file from template...")
            shutil.copy('.env.example', '.env')
            print("✅ .env file created")
            print("⚠️  Please edit .env file with your AWS credentials")
        else:
            print("⚠️  No .env.example found - you may need to configure AWS credentials manually")
    else:
        print("✅ .env file already exists")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    print("✅ Logs directory created")
    
    return True

def check_camera():
    """Check camera availability"""
    print("\n📷 Checking camera availability...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print("✅ Camera is available and working")
                return True
            else:
                print("⚠️  Camera detected but failed to capture frame")
                return False
        else:
            print("❌ No camera detected")
            return False
            
    except ImportError:
        print("❌ OpenCV not installed - cannot check camera")
        return False
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False

def get_python_executable():
    """Get the appropriate Python executable (venv or system)"""
    venv_python = get_venv_python()
    return venv_python if os.path.exists(venv_python) else sys.executable

def run_system_validation():
    """Run system validation"""
    print("\n🧪 Running system validation...")
    
    try:
        python_exe = get_python_executable()
        result = subprocess.run([
            python_exe, 'validate_system.py'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ System validation passed")
            return True
        else:
            print("⚠️  System validation completed with warnings")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return False

def run_tests():
    """Run unit tests"""
    print("\n🧪 Running unit tests...")
    
    try:
        python_exe = get_python_executable()
        result = subprocess.run([
            python_exe, 'tests/test_sleep_detection.py'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("⚠️  Some tests failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def show_menu():
    """Show main menu"""
    venv_status = "🐍 VENV" if is_in_venv() else "🌐 SYSTEM"
    
    print("\n" + "=" * 50)
    print(f"🎯 Sleep Detection System ({venv_status})")
    print("=" * 50)
    print("1. Run Web Interface")
    print("2. Run Command Line System")
    print("3. Run Visual Demo")
    print("4. Run Manual Tests")
    print("5. Run System Validation")
    print("6. Run Unit Tests")
    print("7. Setup Environment & Virtual Environment")
    print("8. Activate Virtual Environment")
    print("0. Exit")
    print("=" * 50)

def run_web_interface():
    """Run the web interface"""
    print("\n🌐 Starting Web Interface...")
    print("The web interface will be available at: http://127.0.0.1:5001")
    print("Press Ctrl+C to stop")
    
    try:
        python_exe = get_python_executable()
        subprocess.run([python_exe, 'sleep_web_ui.py'])
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")

def run_command_line():
    """Run the command line system"""
    print("\n💻 Starting Command Line System...")
    print("Press Ctrl+C to stop")
    
    try:
        python_exe = get_python_executable()
        subprocess.run([python_exe, 'sleep_detection_system.py'])
    except KeyboardInterrupt:
        print("\n🛑 Command line system stopped")

def run_visual_demo():
    """Run the visual demo"""
    print("\n🎬 Starting Visual Demo...")
    
    try:
        python_exe = get_python_executable()
        subprocess.run([python_exe, 'visual_demo.py'])
    except KeyboardInterrupt:
        print("\n🛑 Visual demo stopped")

def run_manual_tests():
    """Run manual test scenarios"""
    print("\n🧪 Starting Manual Tests...")
    
    try:
        python_exe = get_python_executable()
        subprocess.run([python_exe, 'manual_test_scenarios.py'])
    except KeyboardInterrupt:
        print("\n🛑 Manual tests stopped")

def activate_virtual_environment():
    """Show instructions for activating virtual environment"""
    print("\n🐍 Virtual Environment Activation")
    print("=" * 40)
    
    if not os.path.exists('.venv'):
        print("❌ Virtual environment not found. Setting it up first...")
        if setup_virtual_environment():
            print("✅ Virtual environment created")
        else:
            print("❌ Failed to create virtual environment")
            return
    
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
    
    print(f"\nCurrent status: {'🐍 In virtual environment' if is_in_venv() else '🌐 Using system Python'}")
    print("\nAfter activation, you can run scripts directly:")
    print("  python sleep_web_ui.py")
    print("  python sleep_detection_system.py")
    print("  python visual_demo.py")

def main():
    """Main setup and run function"""
    print("🚀 Sleep Detection Module Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check virtual environment status
    if not is_in_venv():
        print(f"\n⚠️  Currently using system Python: {sys.executable}")
        print("💡 Tip: For better isolation, consider using the virtual environment")
        print("   Run option 8 to see activation instructions")
    else:
        print(f"\n✅ Running in virtual environment: {sys.executable}")
    
    # Initial setup check
    setup_needed = (not os.path.exists('.env') or 
                   not os.path.exists('logs') or 
                   not os.path.exists('.venv'))
    
    if setup_needed:
        print("\n🔧 Initial setup required...")
        
        # Setup virtual environment first
        if not os.path.exists('.venv'):
            if not setup_virtual_environment():
                print("❌ Setup failed - could not create virtual environment")
                return
        
        if not install_requirements():
            print("❌ Setup failed - could not install requirements")
            return
        
        if not setup_environment():
            print("❌ Setup failed - could not setup environment")
            return
        
        print("\n✅ Initial setup completed!")
        
        # Optional validation
        response = input("\nRun system validation? (y/n): ").lower().strip()
        if response == 'y':
            run_system_validation()
    
    # Main menu loop
    while True:
        try:
            show_menu()
            choice = input("\nSelect option (0-7): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                run_web_interface()
            elif choice == '2':
                run_command_line()
            elif choice == '3':
                run_visual_demo()
            elif choice == '4':
                run_manual_tests()
            elif choice == '5':
                run_system_validation()
            elif choice == '6':
                run_tests()
            elif choice == '7':
                setup_virtual_environment()
                install_requirements()
                setup_environment()
                print("✅ Environment setup completed")
            elif choice == '8':
                activate_virtual_environment()
            else:
                print("❌ Invalid choice. Please select 0-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    main()