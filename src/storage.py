"""Simple in-memory / SQLite storage for users and documents."""
import os
import sqlite3
import json
from typing import Optional, List, Dict, Any


DB_PATH = os.getenv("DATABASE_URL", "sqlite:///data/app.db").replace("sqlite:///", "")


def _connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def create_user(username: str, password_hash: str) -> int:
    from datetime import datetime
    con = _connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, password_hash, datetime.utcnow().isoformat())
    )
    con.commit()
    uid = cur.lastrowid
    con.close()
    return uid


def get_user(username: str) -> Optional[Dict[str, Any]]:
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    con.close()
    return dict(row) if row else None


def save_document(user_id: int, filename: str, content: str) -> int:
    from datetime import datetime
    con = _connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO documents (user_id, filename, content, uploaded_at) VALUES (?, ?, ?, ?)",
        (user_id, filename, content, datetime.utcnow().isoformat())
    )
    con.commit()
    did = cur.lastrowid
    con.close()
    return did


def list_documents(user_id: int) -> List[Dict[str, Any]]:
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT id, filename, uploaded_at FROM documents WHERE user_id = ? ORDER BY id DESC", (user_id,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def get_documents_content(user_id: int) -> List[Dict[str, Any]]:
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT id, filename, content FROM documents WHERE user_id = ?", (user_id,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def save_chat(user_id: int, question: str, answer: str):
    from datetime import datetime
    con = _connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
        (user_id, question, answer, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()


def get_chat_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    con = _connect()
    cur = con.cursor()
    cur.execute(
        "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows
