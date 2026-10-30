from datetime import datetime
import sqlite3
import pandas as pd

DB_NAME = "digitsense.db"


def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    return conn


def init_db():
    """Create the predictions table with feedback column if not exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_type TEXT,
            predicted_digit INTEGER,
            confidence REAL,
            feedback TEXT DEFAULT 'Pending'
        )
    """
    )
    conn.commit()
    conn.close()


def save_prediction(source_type, digit, confidence):
    """Save a new prediction record and return the generated record ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO predictions (timestamp, source_type, predicted_digit, confidence, feedback)
        VALUES (?, ?, ?, ?, 'Pending')
    """,
        (now, source_type, int(digit), float(confidence)),
    )
    last_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return last_id


def update_feedback(record_id, feedback_value):
    """Update feedback status for a specific prediction ID (Correct / Incorrect)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE predictions
        SET feedback = ?
        WHERE id = ?
    """,
        (feedback_value, record_id),
    )
    conn.commit()
    conn.close()


def fetch_history():
    """Retrieve all prediction records as a pandas DataFrame."""
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC", conn
    )
    conn.close()
    return df


def clear_all_history():
    """Delete all records from the predictions table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()