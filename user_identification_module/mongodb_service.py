import pymongo
from pymongo import MongoClient
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from config import Config
import logging

class MongoDBService:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            # Try with TLS settings for Atlas
            self.client = MongoClient(
                Config.MONGODB_URI, 
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsAllowInvalidCertificates=True  # Allow invalid certificates
            )
            
            # Test connection
            self.client.server_info()
            
            self.db = self.client[Config.MONGODB_DATABASE]
            self.collection = self.db[Config.MONGODB_COLLECTION]
            
            # Create indexes for better performance
            self.collection.create_index("face_id", unique=True)
            self.collection.create_index("name")
            self.collection.create_index("created_at")
            
            self.connected = True
            print("✅ MongoDB Atlas connected successfully")
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("💡 Falling back to local SQLite database")
            self.connected = False
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        return self.connected
    
    def add_user(self, face_id: str, name: str, face_encoding: List[float], data: Dict[Any, Any]) -> bool:
        """Add a new user to MongoDB"""
        if not self.connected:
            return False
        
        try:
            user_doc = {
                'face_id': face_id,
                'name': name,
                'face_encoding': face_encoding,
                'data': data,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            self.collection.insert_one(user_doc)
            return True
            
        except pymongo.errors.DuplicateKeyError:
            print(f"User with face_id {face_id} already exists")
            return False
        except Exception as e:
            print(f"Error adding user to MongoDB: {e}")
            return False
    
    def get_user_by_face_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user data by face ID"""
        if not self.connected:
            return None
        
        try:
            user_doc = self.collection.find_one({'face_id': face_id})
            if user_doc:
                return {
                    'name': user_doc['name'],
                    'data': user_doc['data'],
                    'face_encoding': user_doc['face_encoding']
                }
            return None
        except Exception as e:
            print(f"Error retrieving user from MongoDB: {e}")
            return None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users with their face encodings"""
        if not self.connected:
            return []
        
        try:
            users = list(self.collection.find({}, {
                'face_id': 1, 
                'name': 1, 
                'face_encoding': 1, 
                'data': 1
            }))
            return users
        except Exception as e:
            print(f"Error retrieving users from MongoDB: {e}")
            return []
    
    def list_users(self) -> List[tuple]:
        """List all users (face_id, name) for compatibility with SQLite interface"""
        if not self.connected:
            return []
        
        try:
            users = self.collection.find({}, {'face_id': 1, 'name': 1})
            return [(user['face_id'], user['name']) for user in users]
        except Exception as e:
            print(f"Error listing users from MongoDB: {e}")
            return []
    
    def delete_user(self, face_id: str) -> bool:
        """Delete a user by face_id"""
        if not self.connected:
            return False
        
        try:
            result = self.collection.delete_one({'face_id': face_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting user from MongoDB: {e}")
            return False
    
    def update_user(self, face_id: str, name: str = None, data: Dict[Any, Any] = None) -> bool:
        """Update user information"""
        if not self.connected:
            return False
        
        try:
            update_doc = {'updated_at': datetime.utcnow()}
            if name:
                update_doc['name'] = name
            if data is not None:
                update_doc['data'] = data
            
            result = self.collection.update_one(
                {'face_id': face_id},
                {'$set': update_doc}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating user in MongoDB: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False