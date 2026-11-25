"""SQLite persistence for saved evaluations and business analyses."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get_paths
from .utils import json_safe


def _connect() -> sqlite3.Connection:
    paths = get_paths()
    paths.evaluations_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(paths.evaluations_db)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                title TEXT NOT NULL,
                language TEXT NOT NULL,
                preset TEXT NOT NULL,
                overall_score REAL,
                success_probability REAL,
                roi_percent REAL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_evaluation(report: dict[str, Any]) -> int:
    """Insert a report and return its database id."""
    init_db()
    payload = json_safe(dict(report))
    created_at = datetime.now().isoformat(timespec="seconds")
    financials = report.get("financials") or {}
    mode = str(report.get("mode", "idea"))
    title = str(report.get("title") or report.get("sector_label") or "Evaluation")[:180]
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluations
                (created_at, mode, title, language, preset, overall_score, success_probability, roi_percent, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                mode,
                title,
                str(report.get("language", "en")),
                str(report.get("preset", "tunisia")),
                report.get("overall_score"),
                report.get("success_probability"),
                financials.get("roi_percent"),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        report_id = int(cursor.lastrowid)
        payload["report_id"] = report_id
        connection.execute(
            "UPDATE evaluations SET payload_json = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), report_id),
        )
        connection.commit()
    report["report_id"] = report_id
    return report_id


def list_evaluations(limit: int = 100, mode: str | None = None) -> list[dict[str, Any]]:
    init_db()
    query = (
        "SELECT id, created_at, mode, title, language, preset, overall_score, "
        "success_probability, roi_percent FROM evaluations"
    )
    params: list[Any] = []
    if mode:
        query += " WHERE mode = ?"
        params.append(mode)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_evaluation(report_id: int) -> dict[str, Any] | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            "SELECT payload_json FROM evaluations WHERE id = ?", (int(report_id),)
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["payload_json"])
    except json.JSONDecodeError:
        return None


def delete_evaluation(report_id: int) -> bool:
    init_db()
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM evaluations WHERE id = ?", (int(report_id),))
        connection.commit()
        return cursor.rowcount > 0


def database_path() -> Path:
    return get_paths().evaluations_db
