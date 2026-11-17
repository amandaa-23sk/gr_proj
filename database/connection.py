import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "solar_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "********")
DB_PORT = os.getenv("DB_PORT", "5432")


def get_db_connection():
    """
    Establishes and returns a new connection to the PostgreSQL database.
    """
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )


def init_db():
    """
    Initializes the readings table with telemetry, digital twin, and diagnosis columns.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            temp FLOAT NOT NULL,
            irr FLOAT NOT NULL,
            v FLOAT NOT NULL,
            c FLOAT NOT NULL,
            actual_power FLOAT,
            expected_power FLOAT,
            power_deficit FLOAT,
            performance_ratio FLOAT,
            label VARCHAR(50),
            confidence FLOAT
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("[OK] Database table 'readings' initialized successfully.")


def insert_reading(
    temp: float,
    irr: float,
    v: float,
    c: float,
    actual_power: float = None,
    expected_power: float = None,
    power_deficit: float = None,
    performance_ratio: float = None,
    label: str = None,
    confidence: float = None
) -> dict:
    """
    Inserts a new sensor reading into the readings table and returns the saved record.
    """
    if actual_power is None:
        actual_power = round(float(v) * float(c), 2)

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = """
        INSERT INTO readings (
            temp, irr, v, c,
            actual_power, expected_power, power_deficit, performance_ratio,
            label, confidence
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *;
    """
    cursor.execute(query, (
        float(temp), float(irr), float(v), float(c),
        actual_power, expected_power, power_deficit, performance_ratio,
        label, confidence
    ))
    saved_row = dict(cursor.fetchone())
    conn.commit()
    cursor.close()
    conn.close()
    return saved_row


def fetch_latest_readings(limit: int = 1) -> list:
    """
    Retrieves the most recent sensor reading(s) from the readings table.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = """
        SELECT * FROM readings
        ORDER BY timestamp DESC, id DESC
        LIMIT %s;
    """
    cursor.execute(query, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return rows


if __name__ == "__main__":
    init_db()
