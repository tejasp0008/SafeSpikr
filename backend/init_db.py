# init_db.py
import sqlite3
import os
from pathlib import Path

DB_PATH = "safe_spikr.db"

schema = """
PRAGMA foreign_keys = ON;

-- users table stores a compact face embedding blob and created timestamp
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB,
    created_at REAL
);

-- per-user settings used by backend personalization
CREATE TABLE IF NOT EXISTS user_settings (
    user_id INTEGER PRIMARY KEY,
    drowsy_threshold REAL DEFAULT 0.85,
    distracted_threshold REAL DEFAULT 0.8,
    bias_shift REAL DEFAULT 0.0,
    ewma_alpha REAL DEFAULT 0.2,
    baseline_p_drowsy REAL DEFAULT 0.0,
    last_calibrated REAL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- inference history (raw + personalized)
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    raw_label TEXT,
    raw_confidence REAL,
    personalized_label TEXT,
    personalized_confidence REAL,
    raw_probs TEXT,
    ts REAL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- useful indexes
CREATE INDEX IF NOT EXISTS idx_history_user_ts ON history (user_id, ts DESC);
"""

def main():
    here = Path(__file__).resolve().parent
    db_file = here / DB_PATH
    created = not db_file.exists()
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.executescript(schema)
    conn.commit()
    conn.close()
    if created:
        print(f"Created new DB at: {db_file}")
    else:
        print(f"DB already exists (schema ensured) at: {db_file}")

if __name__ == "__main__":
    main()
