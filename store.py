"""Storage layer: SQLite-backed tasks, idempotency records, decision cache, execution ledger.

Single-process design: one uvicorn worker. SQLite gives durability across restarts;
in-process asyncio locks (see main.py) give atomicity for concurrent requests.
"""
import json
import os
import sqlite3
import threading
import time


def _default_db_path() -> str:
    env = os.environ.get("DB_PATH")
    if env:
        return env
    for d in ("/data", os.path.join(os.path.dirname(__file__), "..", "data")):
        try:
            os.makedirs(d, exist_ok=True)
            test = os.path.join(d, ".w")
            with open(test, "w") as f:
                f.write("x")
            os.remove(test)
            return os.path.join(d, "agent.db")
        except OSError:
            continue
    return "agent.db"


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or _default_db_path()
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks(
              id TEXT PRIMARY KEY, principal TEXT NOT NULL, context_id TEXT NOT NULL,
              state TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS tasks_principal ON tasks(principal, updated);
            CREATE TABLE IF NOT EXISTS messages(
              principal TEXT NOT NULL, message_id TEXT NOT NULL, hash TEXT NOT NULL,
              task_id TEXT NOT NULL, created REAL NOT NULL,
              PRIMARY KEY(principal, message_id));
            CREATE TABLE IF NOT EXISTS decisions(
              key TEXT PRIMARY KEY, body TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS executions(
              principal TEXT NOT NULL, action_id TEXT NOT NULL, task_id TEXT NOT NULL,
              body TEXT NOT NULL, created REAL NOT NULL,
              PRIMARY KEY(principal, action_id));
            CREATE TABLE IF NOT EXISTS inbound(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, principal TEXT,
              kind TEXT, task_id TEXT, body TEXT);
            """
        )

    # ---------- tasks ----------
    def get_task(self, principal: str, task_id: str):
        with self._lock:
            row = self.db.execute(
                "SELECT body FROM tasks WHERE id=? AND principal=?", (task_id, principal)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_tasks(self, principal: str, context_id=None, state=None):
        q = "SELECT body FROM tasks WHERE principal=?"
        args = [principal]
        if context_id:
            q += " AND context_id=?"
            args.append(context_id)
        if state:
            q += " AND state=?"
            args.append(state)
        q += " ORDER BY updated DESC, id"
        with self._lock:
            rows = self.db.execute(q, args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def message_record(self, principal: str, message_id: str):
        with self._lock:
            row = self.db.execute(
                "SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?",
                (principal, message_id),
            ).fetchone()
        return row  # (hash, task_id) or None

    def save_task(self, principal: str, task: dict, message_id: str | None = None,
                  message_hash: str | None = None, executions: list | None = None,
                  create: bool = False):
        """Atomically write the task, the idempotency record and executed actions."""
        now = time.time()
        with self._lock:
            cur = self.db.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                if create:
                    cur.execute(
                        "INSERT INTO tasks(id, principal, context_id, state, body, created, updated)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (task["id"], principal, task["contextId"], task["status"]["state"],
                         dumps(task), now, now),
                    )
                else:
                    cur.execute(
                        "UPDATE tasks SET state=?, body=?, updated=? WHERE id=? AND principal=?",
                        (task["status"]["state"], dumps(task), now, task["id"], principal),
                    )
                if message_id is not None:
                    cur.execute(
                        "INSERT INTO messages(principal, message_id, hash, task_id, created)"
                        " VALUES(?,?,?,?,?)",
                        (principal, message_id, message_hash, task["id"], now),
                    )
                for ex in executions or []:
                    # Ledger keyed by actionId: an action can be executed at most once.
                    cur.execute(
                        "INSERT OR IGNORE INTO executions(principal, action_id, task_id, body, created)"
                        " VALUES(?,?,?,?,?)",
                        (principal, ex["actionId"], task["id"], dumps(ex), now),
                    )
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def executed(self, principal: str, action_id: str):
        with self._lock:
            row = self.db.execute(
                "SELECT body FROM executions WHERE principal=? AND action_id=?",
                (principal, action_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    # ---------- decision cache ----------
    def get_decision(self, key: str):
        with self._lock:
            row = self.db.execute("SELECT body FROM decisions WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_decision(self, key: str, decision: dict):
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO decisions(key, body, created) VALUES(?,?,?)",
                (key, dumps(decision), time.time()),
            )

    def clear_decisions(self):
        with self._lock:
            self.db.execute("DELETE FROM decisions")

    # ---------- debug log ----------
    def log_inbound(self, principal: str, kind: str, task_id: str | None, body):
        try:
            with self._lock:
                self.db.execute(
                    "INSERT INTO inbound(ts, principal, kind, task_id, body) VALUES(?,?,?,?,?)",
                    (time.time(), principal[:12], kind, task_id, dumps(body)),
                )
                self.db.execute(
                    "DELETE FROM inbound WHERE id < (SELECT MAX(id) - 400 FROM inbound)"
                )
        except Exception:
            pass

    def recent_inbound(self, limit: int = 20, kind: str | None = None):
        q = "SELECT id, ts, principal, kind, task_id, body FROM inbound"
        args = []
        if kind:
            q += " WHERE kind=?"
            args.append(kind)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.db.execute(q, args).fetchall()
        return [
            {"id": r[0], "ts": r[1], "principal": r[2], "kind": r[3], "taskId": r[4],
             "body": json.loads(r[5])}
            for r in rows
        ]
