"""SQLite persistence for the local Solar Inspection workspace."""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
 id INTEGER PRIMARY KEY, name TEXT NOT NULL, company TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inspections (
 id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
 title TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT,
 settings_json TEXT NOT NULL, snapshot_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
 id TEXT NOT NULL, inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
 original_name TEXT NOT NULL, image_path TEXT NOT NULL, width INTEGER, height INTEGER,
 PRIMARY KEY (inspection_id, id)
);
CREATE TABLE IF NOT EXISTS findings (
 inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
 cell_index INTEGER NOT NULL, item_id TEXT, cell_row INTEGER, cell_col INTEGER,
 prediction TEXT NOT NULL, confidence REAL, dsi REAL, payload_json TEXT NOT NULL,
 PRIMARY KEY (inspection_id, cell_index),
 FOREIGN KEY (inspection_id, item_id) REFERENCES items(inspection_id, id)
);
CREATE TABLE IF NOT EXISTS activity_logs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id),
 inspection_id TEXT REFERENCES inspections(id) ON DELETE SET NULL,
 event TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS activity_date ON activity_logs(created_at);
CREATE INDEX IF NOT EXISTS inspections_date ON inspections(created_at);
PRAGMA user_version = 1;
"""


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection(root):
    Path(root).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(Path(root) / "solar_inspection.sqlite3", timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    db.executescript(SCHEMA)
    db.execute("INSERT OR IGNORE INTO users VALUES (1, 'Local operator', '', ?)", (timestamp(),))
    try:
        with db:
            yield db
    finally:
        db.close()


def profile(root):
    with connection(root) as db:
        return dict(db.execute("SELECT id, name, company FROM users WHERE id=1").fetchone())


def update_profile(root, data):
    name = str(data.get("name", "")).strip()
    company = str(data.get("company", "")).strip()
    if not name or len(name) > 80 or len(company) > 100:
        raise ValueError("Enter an operator name (up to 80 characters) and business name (up to 100).")
    if any(ord(c) < 32 for c in name + company):
        raise ValueError("Names cannot contain control characters.")
    with connection(root) as db:
        db.execute("UPDATE users SET name=?, company=? WHERE id=1", (name, company))
        db.execute("INSERT INTO activity_logs(user_id,event,detail,created_at) VALUES(1,'profile_updated',?,?)",
                   ("Business profile updated", timestamp()))
    return profile(root)


def save_job(root, job):
    with connection(root) as db:
        previous = db.execute("SELECT status FROM inspections WHERE id=?", (job["id"],)).fetchone()
        title = job.get("title") or (job["sources"][0]["name"] if job["sources"] else "New inspection")
        db.execute("""INSERT INTO inspections VALUES (?,1,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET title=excluded.title,status=excluded.status,
            finished_at=excluded.finished_at,settings_json=excluded.settings_json,snapshot_json=excluded.snapshot_json""",
                   (job["id"], title, job["status"], job["created"], job.get("finished"),
                    json.dumps(job["settings"]), json.dumps(job)))
        for source in job["sources"]:
            cursor = db.execute("INSERT OR IGNORE INTO items VALUES (?,?,?,?,?,?)",
                               (source["id"], job["id"], source["name"], source["path"], source["width"], source["height"]))
            if cursor.rowcount:
                db.execute("INSERT INTO activity_logs(user_id,inspection_id,event,detail,created_at) VALUES(1,?,'image_uploaded',?,?)",
                           (job["id"], source["name"], timestamp()))
        saved_count = db.execute("SELECT COUNT(*) FROM findings WHERE inspection_id=?", (job["id"],)).fetchone()[0]
        for index, result in enumerate(job["results"][saved_count:], start=saved_count):
            db.execute("INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?)",
                       (job["id"], index, result.get("source_id"), result.get("row"), result.get("col"),
                        result.get("prediction", "ERROR"), result.get("confidence"), result.get("dsi"), json.dumps(result)))
        if previous is None or previous["status"] != job["status"]:
            db.execute("INSERT INTO activity_logs(user_id,inspection_id,event,detail,created_at) VALUES(1,?,?,?,?)",
                       (job["id"], "inspection_" + job["status"], title, timestamp()))


def load_jobs(root):
    with connection(root) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT snapshot_json FROM inspections")]


def record_event(root, job_id, event, detail):
    with connection(root) as db:
        db.execute("INSERT INTO activity_logs(user_id,inspection_id,event,detail,created_at) VALUES(1,?,?,?,?)",
                   (job_id, event, detail, timestamp()))


def delete_job(root, job_id):
    with connection(root) as db:
        db.execute("INSERT INTO activity_logs(user_id,inspection_id,event,detail,created_at) VALUES(1,?,'inspection_deleted',?,?)",
                   (job_id, "Deleted inspection " + job_id[:8], timestamp()))
        db.execute("DELETE FROM inspections WHERE id=?", (job_id,))


def dashboard(root):
    with connection(root) as db:
        totals = dict(db.execute("""SELECT COUNT(*) AS inspections,
          COALESCE(SUM(status IN ('queued','running','uploading')),0) AS active,
          COALESCE(SUM(status='completed'),0) AS completed FROM inspections""").fetchone())
        totals.update(dict(db.execute("""SELECT COUNT(*) AS cells,
          COALESCE(SUM(prediction IN ('UNCERTAIN','REJECTED','ERROR')),0) AS review,
          COALESCE(SUM(prediction NOT IN ('normal','UNCERTAIN','REJECTED','ERROR')),0) AS defects,
          COALESCE(SUM(prediction='normal'),0) AS normal FROM findings""").fetchone()))
        recent = [dict(row) for row in db.execute("""SELECT i.id,i.title,i.status,i.created_at,
          (SELECT COUNT(*) FROM findings f WHERE f.inspection_id=i.id) AS cells,
          (SELECT COUNT(*) FROM findings f WHERE f.inspection_id=i.id AND f.prediction IN ('UNCERTAIN','REJECTED','ERROR')) AS review
          FROM inspections i ORDER BY created_at DESC LIMIT 8""")]
        activity = [dict(row) for row in db.execute("SELECT * FROM activity_logs ORDER BY id DESC LIMIT 12")]
        days = [(datetime.now(timezone.utc).date() - timedelta(days=n)).isoformat() for n in range(6, -1, -1)]
        counts = dict(db.execute("SELECT substr(created_at,1,10),COUNT(*) FROM inspections WHERE created_at>=? GROUP BY substr(created_at,1,10)", (days[0],)).fetchall())
        distribution = [dict(row) for row in db.execute("SELECT prediction,COUNT(*) AS count FROM findings GROUP BY prediction ORDER BY count DESC")]
    return dict(totals=totals, recent=recent, activity=activity,
                trend=[{"date": day, "count": counts.get(day, 0)} for day in days], distribution=distribution)
