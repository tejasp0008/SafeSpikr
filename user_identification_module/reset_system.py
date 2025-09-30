#!/usr/bin/env python3
"""
Reset Face Recognition System
Clears all user data from both local database and AWS Rekognition
"""

import os
import sqlite3
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

def reset_sqlite_database():
    """Reset the local SQLite database"""
    print("🗑️ Resetting SQLite database...")
    
    db_path = 'users.db'
    
    try:
        if os.path.exists(db_path):
            # Delete the database file
            os.remove(db_path)
            print(f"✅ Deleted database file: {db_path}")
        else:
            print(f"ℹ️ Database file {db_path} doesn't exist")
        
        # Recreate empty database
        from database import UserDatabase
        db = UserDatabase()
        print("✅ Created new empty database")
        
        return True
        
    except Exception as e:
        print(f"❌ Error resetting SQLite database: {e}")
        return False

def reset_aws_rekognition():
    """Reset AWS Rekognition collection"""
    print("\n🗑️ Resetting AWS Rekognition collection...")
    
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
        
        # Check if collection exists
        try:
            response = rekognition.describe_collection(CollectionId=collection_id)
            face_count = response.get('FaceCount', 0)
            print(f"📊 Found collection '{collection_id}' with {face_count} faces")
            
            if face_count > 0:
                # List all faces in the collection
                print("🔍 Listing all faces in collection...")
                faces_response = rekognition.list_faces(CollectionId=collection_id)
                face_ids = [face['FaceId'] for face in faces_response.get('Faces', [])]
                
                if face_ids:
                    print(f"🗑️ Deleting {len(face_ids)} faces...")
                    
                    # Delete faces in batches (AWS limit is 4096 per request)
                    batch_size = 4096
                    for i in range(0, len(face_ids), batch_size):
                        batch = face_ids[i:i + batch_size]
                        rekognition.delete_faces(
                            CollectionId=collection_id,
                            FaceIds=batch
                        )
                        print(f"✅ Deleted batch of {len(batch)} faces")
                    
                    print(f"✅ All faces deleted from collection '{collection_id}'")
                else:
                    print("ℹ️ No faces found in collection")
            else:
                print("ℹ️ Collection is already empty")
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"ℹ️ Collection '{collection_id}' doesn't exist")
                
                # Create new empty collection
                print(f"📦 Creating new collection '{collection_id}'...")
                rekognition.create_collection(CollectionId=collection_id)
                print(f"✅ Created new empty collection '{collection_id}'")
                return True
            else:
                raise e
                
    except Exception as e:
        print(f"❌ Error resetting AWS Rekognition: {e}")
        return False

def reset_mongodb():
    """Reset MongoDB collection (if used)"""
    print("\n🗑️ Checking MongoDB...")
    
    try:
        from mongodb_service import MongoDBService
        mongo = MongoDBService()
        
        if mongo.is_connected():
            # Get collection
            collection = mongo.collection
            
            # Count documents
            doc_count = collection.count_documents({})
            print(f"📊 Found {doc_count} documents in MongoDB")
            
            if doc_count > 0:
                # Delete all documents
                result = collection.delete_many({})
                print(f"✅ Deleted {result.deleted_count} documents from MongoDB")
            else:
                print("ℹ️ MongoDB collection is already empty")
            
            mongo.close()
            return True
        else:
            print("ℹ️ MongoDB not connected, skipping")
            return True
            
    except Exception as e:
        print(f"⚠️ MongoDB reset error (non-critical): {e}")
        return True  # Non-critical error

def clean_debug_files():
    """Clean up debug files"""
    print("\n🧹 Cleaning debug files...")
    
    debug_patterns = [
        'debug_frame_*.jpg',
        'debug_face_detection.jpg'
    ]
    
    import glob
    
    cleaned_count = 0
    for pattern in debug_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"🗑️ Deleted: {file}")
                cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Could not delete {file}: {e}")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned {cleaned_count} debug files")
    else:
        print("ℹ️ No debug files to clean")

def main():
    """Main reset function"""
    print("🚀 Face Recognition System Reset")
    print("=" * 50)
    print("⚠️ WARNING: This will delete ALL user data!")
    print("   - Local SQLite database")
    print("   - AWS Rekognition faces")
    print("   - MongoDB data (if connected)")
    print("   - Debug files")
    print("=" * 50)
    
    # Confirmation
    confirm = input("Are you sure you want to reset everything? (type 'YES' to confirm): ")
    
    if confirm != 'YES':
        print("❌ Reset cancelled")
        return
    
    print("\n🗑️ Starting system reset...")
    
    # Reset components
    results = {}
    
    print("\n" + "="*30 + " RESET PROCESS " + "="*30)
    
    results['sqlite'] = reset_sqlite_database()
    results['aws'] = reset_aws_rekognition()
    results['mongodb'] = reset_mongodb()
    
    print("\n" + "="*30 + " CLEANUP " + "="*30)
    clean_debug_files()
    
    # Summary
    print("\n" + "="*30 + " SUMMARY " + "="*30)
    print("📊 Reset Results:")
    
    for component, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {component.upper()}: {status}")
    
    if all(results.values()):
        print("\n🎉 System reset completed successfully!")
        print("💡 You can now start fresh with:")
        print("   python aws_web_ui.py")
    else:
        print("\n⚠️ Some components failed to reset")
        print("💡 Check the errors above and try again")

if __name__ == "__main__":
    main()