#!/usr/bin/env python3
"""
Test MongoDB Atlas connection
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import ssl

def test_mongodb_connection():
    """Test different MongoDB connection methods"""
    load_dotenv(override=True)
    
    mongodb_uri = os.getenv('MONGODB_URI')
    print(f"🔍 Testing MongoDB connection...")
    print(f"URI: {mongodb_uri[:50]}...")
    
    # Method 1: With SSL certificate validation disabled
    print("\n📋 Method 1: TLS validation disabled")
    try:
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            tlsAllowInvalidCertificates=True,
            tlsInsecure=True
        )
        
        # Test connection
        info = client.server_info()
        print("✅ Connection successful!")
        print(f"   MongoDB version: {info.get('version')}")
        
        # Test database operations
        db = client[os.getenv('MONGODB_DATABASE', 'face_recognition')]
        collection = db[os.getenv('MONGODB_COLLECTION', 'users')]
        
        # Insert test document
        test_doc = {"test": True, "message": "Connection test"}
        result = collection.insert_one(test_doc)
        print(f"✅ Test document inserted: {result.inserted_id}")
        
        # Delete test document
        collection.delete_one({"_id": result.inserted_id})
        print("✅ Test document deleted")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: With different TLS settings
    print("\n📋 Method 2: Alternative TLS settings")
    try:
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsAllowInvalidCertificates=True
        )
        
        info = client.server_info()
        print("✅ Alternative method successful!")
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Using the service class
    print("\n📋 Method 3: Using MongoDB service class")
    try:
        from mongodb_service import MongoDBService
        mongo = MongoDBService()
        
        if mongo.is_connected():
            print("✅ MongoDB service connected successfully!")
            
            # Test operations
            test_users = mongo.get_all_users()
            print(f"✅ Found {len(test_users)} existing users")
            
            mongo.close()
            return True
        else:
            print("❌ MongoDB service failed to connect")
            
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    return False

def main():
    """Main test function"""
    print("🚀 MongoDB Atlas Connection Test")
    print("=" * 50)
    
    success = test_mongodb_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 MongoDB Atlas connection working!")
        print("💡 The fallback system should now work properly")
    else:
        print("❌ MongoDB Atlas connection failed")
        print("💡 System will fall back to SQLite database")
        print("\n🔧 Possible solutions:")
        print("1. Check your MongoDB Atlas credentials")
        print("2. Ensure your IP is whitelisted in MongoDB Atlas")
        print("3. Verify the connection string is correct")

if __name__ == "__main__":
    main()