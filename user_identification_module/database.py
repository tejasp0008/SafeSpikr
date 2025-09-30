import sqlite3
import json
from typing import Optional, Dict, Any
from config import Config

class UserDatabase:
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the database with users table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_user(self, face_id: str, name: str, data: Dict[Any, Any]) -> bool:
        """Add a new user to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO users (face_id, name, data) VALUES (?, ?, ?)',
                    (face_id, name, json.dumps(data))
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
    
    def get_user_by_face_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user data by face ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT name, data FROM users WHERE face_id = ?',
                (face_id,)
            )
            result = cursor.fetchone()
            
            if result:
                name, data_json = result
                return {
                    'name': name,
                    'data': json.loads(data_json)
                }
            return None
    
    def list_users(self) -> list:
        """List all users in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT face_id, name FROM users')
            return cursor.fetchall()