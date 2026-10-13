import sqlite3
from pathlib import Path

DB_PATH = Path("models/chat_memory.db")

class ChatDatabase:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS variables (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Session-scoped store for the active attached document, split into
        # coherent sections so follow-up questions can be answered from it.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_index INTEGER PRIMARY KEY,
                content TEXT
            )
        """)
        self.conn.commit()

    def add_message(self, role, content):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        self.conn.commit()

    def get_messages(self, limit=6):
        cursor = self.conn.cursor()
        cursor.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def delete_last_message(self, role=None):
        """Delete the most recent message (optionally only if it matches role)."""
        cursor = self.conn.cursor()
        if role is None:
            cursor.execute("DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages)")
        else:
            cursor.execute(
                "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages WHERE role = ?)",
                (role,),
            )
        deleted = cursor.rowcount
        self.conn.commit()
        return deleted

    def set_variable(self, key, value):
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO variables (key, value) VALUES (?, ?)", (key, str(value)))
        self.conn.commit()

    def get_variable(self, key):
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM variables WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_all_variables(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM variables")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def set_document(self, filename, chunks):
        """Store the active document's sections, replacing any previous one."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM document_chunks")
        cursor.executemany(
            "INSERT INTO document_chunks (chunk_index, content) VALUES (?, ?)",
            [(i, c) for i, c in enumerate(chunks)],
        )
        cursor.execute(
            "INSERT OR REPLACE INTO variables (key, value) VALUES ('active_document', ?)",
            (str(filename),),
        )
        self.conn.commit()

    def get_document_chunks(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT content FROM document_chunks ORDER BY chunk_index ASC")
        return [row[0] for row in cursor.fetchall()]

    def get_active_document_name(self):
        return self.get_variable("active_document")

    def has_document(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        return cursor.fetchone()[0] > 0

    def clear_document(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM document_chunks")
        cursor.execute("DELETE FROM variables WHERE key = 'active_document'")
        self.conn.commit()

    def clear(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM variables")
        cursor.execute("DELETE FROM document_chunks")
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
        if DB_PATH.exists():
            try:
                DB_PATH.unlink()
            except Exception:
                pass
