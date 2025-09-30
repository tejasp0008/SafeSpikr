#!/usr/bin/env python3
"""
Face Recognition System using AWS Rekognition
Main entry point for the application
"""

import sys
from face_recognition_system import FaceRecognitionSystem

def print_menu():
    """Print the main menu"""
    print("\n" + "="*50)
    print("Face Recognition System")
    print("="*50)
    print("1. Start Camera & Recognition")
    print("2. List Registered Users")
    print("3. Exit")
    print("="*50)

def main():
    """Main application entry point"""
    system = FaceRecognitionSystem()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nStarting camera system...")
            print("Controls:")
            print("- Press 's' to scan for existing user")
            print("- Press 'a' to add new user")
            print("- Press 'q' to return to main menu")
            system.start_system()
            
        elif choice == '2':
            system.list_users()
            
        elif choice == '3':
            print("Goodbye!")
            sys.exit(0)
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()