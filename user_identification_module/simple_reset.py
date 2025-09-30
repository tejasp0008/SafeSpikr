#!/usr/bin/env python3
"""
Simple Reset - Just delete local files
No AWS interaction required
"""

import os
import glob

def simple_reset():
    """Simple reset that only touches local files"""
    print("🗑️ Simple Local Reset")
    print("=" * 30)
    
    files_deleted = 0
    
    # Delete SQLite database
    if os.path.exists('users.db'):
        os.remove('users.db')
        print("✅ Deleted: users.db")
        files_deleted += 1
    
    # Delete debug files
    debug_files = glob.glob('debug_frame_*.jpg') + glob.glob('debug_face_detection.jpg')
    for file in debug_files:
        try:
            os.remove(file)
            print(f"✅ Deleted: {file}")
            files_deleted += 1
        except:
            pass
    
    if files_deleted == 0:
        print("ℹ️ No local files to delete")
    else:
        print(f"✅ Deleted {files_deleted} local files")
    
    # Recreate empty database
    try:
        from database import UserDatabase
        db = UserDatabase()
        print("✅ Created new empty database")
    except Exception as e:
        print(f"⚠️ Could not recreate database: {e}")
    
    print("\n💡 Note: This only resets local data.")
    print("   AWS Rekognition faces are NOT deleted.")
    print("   Use the AWS Console to manually delete the collection if needed.")

if __name__ == "__main__":
    confirm = input("Reset local data only? (y/N): ")
    if confirm.lower() == 'y':
        simple_reset()
    else:
        print("❌ Cancelled")