from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    phone_number TEXT PRIMARY KEY,
    name TEXT,
    phone_tail TEXT,
    lifecycle_stage TEXT NOT NULL DEFAULT 'new',
    preferred_topic TEXT,
    last_intent INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(phone_number) REFERENCES users(phone_number)
);

CREATE TABLE IF NOT EXISTS vector_documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL CHECK(collection IN ('market', 'course')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    vector_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_phone ON chat_messages(phone_number, id);
CREATE INDEX IF NOT EXISTS idx_vector_collection ON vector_documents(collection);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._ensure_column(connection, "users", "phone_tail", "TEXT")

    def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def ensure_user(
        self,
        phone_number: str,
        name: str | None = None,
        phone_tail: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE phone_number = ?", (phone_number,)
            ).fetchone()
            if row:
                if name or phone_tail:
                    connection.execute(
                        """
                        UPDATE users
                        SET name = COALESCE(?, name),
                            phone_tail = COALESCE(?, phone_tail),
                            updated_at = ?
                        WHERE phone_number = ?
                        """,
                        (name, phone_tail, utc_now(), phone_number),
                    )
                    row = connection.execute(
                        "SELECT * FROM users WHERE phone_number = ?", (phone_number,)
                    ).fetchone()
                return dict(row), False
            now = utc_now()
            connection.execute(
                "INSERT INTO users(phone_number, name, phone_tail, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (phone_number, name, phone_tail, now, now),
            )
            row = connection.execute(
                "SELECT * FROM users WHERE phone_number = ?", (phone_number,)
            ).fetchone()
            return dict(row), True

    def get_user(self, phone_number: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE phone_number = ?", (phone_number,)
            ).fetchone()
            return dict(row) if row else None

    def update_user_intent(self, phone_number: str, intent_code: int) -> None:
        preferred_topic = {
            1: "exercise_recommendation",
            2: "training_goal_plan",
            3: "health_assessment",
        }.get(intent_code)
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET last_intent = ?,
                    preferred_topic = COALESCE(?, preferred_topic),
                    lifecycle_stage = CASE WHEN lifecycle_stage = 'new' THEN 'engaged' ELSE lifecycle_stage END,
                    updated_at = ?
                WHERE phone_number = ?
                """,
                (intent_code, preferred_topic, utc_now(), phone_number),
            )

    def fetch_chat_history(self, phone_number: str, limit: int = 12) -> list[dict[str, str]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE phone_number = ? ORDER BY id DESC LIMIT ?
                """,
                (phone_number, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def append_exchange(self, phone_number: str, query: str, response: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO chat_messages(phone_number, role, content, created_at) VALUES (?, ?, ?, ?)",
                [
                    (phone_number, "user", query, now),
                    (phone_number, "assistant", response, now),
                ],
            )

    def upsert_document(
        self,
        document_id: str,
        collection: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        vector: list[float],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO vector_documents(
                    id, collection, title, content, metadata_json, vector_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    collection = excluded.collection,
                    title = excluded.title,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    vector_json = excluded.vector_json,
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    collection,
                    title,
                    content,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(vector),
                    utc_now(),
                ),
            )

    def delete_seed_documents(self) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM vector_documents WHERE metadata_json LIKE ?",
                ('%"source": "seed"%',),
            )

    def list_documents(self, collection: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vector_documents WHERE collection = ?", (collection,)
            ).fetchall()
        documents = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            item["vector"] = json.loads(item.pop("vector_json"))
            documents.append(item)
        return documents
