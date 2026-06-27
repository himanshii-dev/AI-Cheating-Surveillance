"""
database.py
-----------
Handles all SQLite database operations for the AI Cheating Surveillance system.
Creates the violations table and provides functions to insert and fetch records.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database.db")


def get_connection():
    """Return a new SQLite connection to the application database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the database by creating the violations table if it does not exist.
    Called once when the Flask application starts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            violation_type  TEXT    NOT NULL,
            score           INTEGER NOT NULL,
            screenshot_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_violation(violation_type: str, score: int, screenshot_path: str = None):
    """
    Insert a single violation record into the database.

    Parameters
    ----------
    violation_type  : str  – Human-readable label (e.g. "Phone Detected").
    score           : int  – Suspicion score at the time of the violation.
    screenshot_path : str  – Relative path to the saved screenshot, or None.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO violations (timestamp, violation_type, score, screenshot_path)
        VALUES (?, ?, ?, ?)
        """,
        (timestamp, violation_type, score, screenshot_path),
    )
    conn.commit()
    conn.close()


def fetch_recent_violations(limit: int = 20):
    """
    Fetch the most recent violations from the database.

    Parameters
    ----------
    limit : int – Maximum number of rows to return (default 20).

    Returns
    -------
    list[sqlite3.Row] – List of violation rows ordered newest first.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, violation_type, score, screenshot_path
        FROM violations
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_all_violations():
    """
    Fetch every violation from the database, newest first.

    Returns
    -------
    list[sqlite3.Row]
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, violation_type, score, screenshot_path
        FROM violations
        ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
