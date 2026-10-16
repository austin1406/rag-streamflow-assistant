"""Tools the agent can call.

Every tool validates its own arguments and returns {"ok": bool, ...} instead of raising on
bad model input, so the agent loop can feed the error back to the model and let it retry.
"""
import re
import sqlite3
import time
from pathlib import Path

EVAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "eval.sqlite"

MAX_TOP_K = 10
MAX_SQL_ROWS = 50
MAX_SQL_CHARS = 2000
SQL_TIME_LIMIT_SECONDS = 2.0

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)
_ALLOWED_SQLITE_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def search_documents(args: dict, retrieve_fn=None) -> dict:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "query must be a non-empty string"}

    top_k = args.get("top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        return {"ok": False, "error": f"top_k must be an integer from 1 to {MAX_TOP_K}"}

    if retrieve_fn is None:
        from .rag import retrieve as retrieve_fn

    chunks = retrieve_fn(query.strip(), top_k=top_k)
    return {
        "ok": True,
        "chunks": [
            {"id": c["id"], "source": c["source"], "loc": c["loc"], "text": c["text"]}
            for c in chunks
        ],
    }


def _validate_select(sql) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("sql must be a non-empty string")
    statement = sql.strip().rstrip(";").strip()
    if ";" in statement:
        raise ValueError("only a single SQL statement is allowed")
    if not re.match(r"select\b", statement, re.IGNORECASE):
        raise ValueError("only SELECT statements are allowed")
    if _WRITE_KEYWORDS.search(statement):
        raise ValueError("this query uses a keyword that is not allowed (read-only access)")
    return statement


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED_SQLITE_ACTIONS else sqlite3.SQLITE_DENY


def query_eval_results(args: dict, db_path=None) -> dict:
    try:
        statement = _validate_select(args.get("sql"))
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    path = Path(db_path or EVAL_DB_PATH)
    if not path.exists():
        return {"ok": False, "error": "evaluation database not found; run `python -m rag.eval_db`"}

    # Three independent layers: keyword check above, a read-only connection, and a
    # SQLite authorizer that denies everything except reading, plus a time limit.
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.set_authorizer(_authorizer)
        deadline = time.monotonic() + SQL_TIME_LIMIT_SECONDS
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        cursor = conn.execute(statement)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
    except sqlite3.Error as e:
        return {"ok": False, "error": f"SQL error: {e}"}
    finally:
        conn.close()

    truncated = len(rows) > MAX_SQL_ROWS
    rows = [[_clip(v) for v in row] for row in rows[:MAX_SQL_ROWS]]
    return {"ok": True, "columns": columns, "rows": rows, "truncated": truncated}


def _clip(value):
    if isinstance(value, str) and len(value) > MAX_SQL_CHARS:
        return value[:MAX_SQL_CHARS] + "..."
    return value
