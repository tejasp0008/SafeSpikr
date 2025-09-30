#!/usr/bin/env python3
"""
Quick Reset Options
Choose what to reset individually
"""

import os
import sys

def reset_sqlite_only():
    """Reset only SQLite database"""
    print("🗑️ Resetting SQLite database only...")
    
    db_path = 'users.db'
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Deleted: {db_path}")
    
    # Recreate empty database
    from database import UserDatabase
    db = UserDatabase()
    print("✅ Created new empty SQLite database")

def reset_aws_only():
    """Reset only AWS Rekognition collection"""
    print("🗑️ Resetting AWS Rekognition only...")
    
    import boto3
    from botocore.exceptions import ClientError
    from dotenv import load_dotenv
    
    load_dotenv(override=True)
    collection_id = os.getenv('REKOGNITION_COLLECTION_ID', 'face_collection')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    try:
        rekognition = boto3.client(
            'rekognition',
            region_name=aws_region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # List and delete all faces
        faces_response = rekognition.list_faces(CollectionId=collection_id)
        face_ids = [face['FaceId'] for face in faces_response.get('Faces', [])]
        
        if face_ids:
            rekognition.delete_faces(CollectionId=collection_id, FaceIds=face_ids)
            print(f"✅ Deleted {len(face_ids)} faces from AWS collection")
        else:
            print("ℹ️ AWS collection is already empty")
            
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"ℹ️ Collection '{collection_id}' doesn't exist, creating new one...")
            rekognition.create_collection(CollectionId=collection_id)
            print(f"✅ Created new collection '{collection_id}'")
        else:
            print(f"❌ AWS error: {e}")

def show_current_data():
    """Show current data in system"""
    print("📊 Current System Data:")
    print("=" * 30)
    
    # SQLite data
    try:
        from database import UserDatabase
        db = UserDatabase()
        users = db.list_users()
        print(f"SQLite Users: {len(users)}")
        for face_id, name in users:
            print(f"  - {name} (ID: {face_id[:8]}...)")
    except Exception as e:
        print(f"SQLite: Error - {e}")
    
    # AWS data
    try:
        import boto3
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        collection_id = os.getenv('REKOGNITION_COLLECTION_ID', 'face_collection')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        rekognition = boto3.client(
            'rekognition',
            region_name=aws_region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        response = rekognition.describe_collection(CollectionId=collection_id)
        face_count = response.get('FaceCount', 0)
        print(f"AWS Faces: {face_count}")
        
        if face_count > 0:
            faces_response = rekognition.list_faces(CollectionId=collection_id, MaxResults=10)
            for face in faces_response.get('Faces', []):
                face_id = face['FaceId']
                print(f"  - Face ID: {face_id[:8]}...")
                
    except Exception as e:
        print(f"AWS: Error - {e}")

def main():
    """Main menu"""
    while True:
        print("\n🔧 Quick Reset Options")
        print("=" * 30)
        print("1. Show current data")
        print("2. Reset SQLite database only")
        print("3. Reset AWS Rekognition only")
        print("4. Reset everything (run full reset)")
        print("5. Exit")
        print("=" * 30)
        
        choice = input("Choose option (1-5): ").strip()
        
        if choice == '1':
            show_current_data()
            
        elif choice == '2':
            confirm = input("Reset SQLite database? (y/N): ")
            if confirm.lower() == 'y':
                reset_sqlite_only()
            else:
                print("❌ Cancelled")
                
        elif choice == '3':
            confirm = input("Reset AWS Rekognition collection? (y/N): ")
            if confirm.lower() == 'y':
                reset_aws_only()
            else:
                print("❌ Cancelled")
                
        elif choice == '4':
            print("🚀 Running full system reset...")
            os.system('python reset_system.py')
            
        elif choice == '5':
            print("👋 Goodbye!")
            sys.exit(0)
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()