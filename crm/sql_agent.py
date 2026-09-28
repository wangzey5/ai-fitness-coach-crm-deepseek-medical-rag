from __future__ import annotations

import re
import sqlite3
from typing import Any

from .database import Database


DISALLOWED_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


class ReadOnlySQLAgent:
    """Small SQL agent boundary that only permits one parameterized SELECT statement."""

    def __init__(self, database: Database):
        self.database = database

    def execute(self, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
        statement = sql.strip().rstrip(";")
        if not statement.lower().startswith("select ") or DISALLOWED_SQL.search(statement):
            raise ValueError("Only read-only SELECT queries are allowed")
        if ";" in statement:
            raise ValueError("Only one SQL statement is allowed")
        uri = f"file:{self.database.path.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(statement, parameters or []).fetchmany(100)
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def user_summary(self, phone_number: str) -> dict[str, Any] | None:
        rows = self.execute(
            """
            SELECT u.phone_number, u.name, u.lifecycle_stage, u.preferred_topic,
                   u.last_intent, COUNT(m.id) AS message_count
            FROM users u
            LEFT JOIN chat_messages m ON m.phone_number = u.phone_number
            WHERE u.phone_number = ?
            GROUP BY u.phone_number
            """,
            [phone_number],
        )
        return rows[0] if rows else None

