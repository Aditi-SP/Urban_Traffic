"""
Community Reports: a small, local, anonymous store of short text reports (SQLite from the standard library; no new dependency).

What this is
  * Reports are typed by road users, are ANONYMOUS and UNVERIFIED, and are shown as exactly that. They are not traffic measurements
    and are never used as model input, training data or evidence of an incident.
  * They are stored on THIS computer, in data/community/community.db (override with the COMMUNITY_DB_PATH environment variable).
    That folder is git-ignored. Anyone using the same running dashboard sees the same reports.
  * Nothing here makes a network request. Photo reports (with GPS and a 10-minute freshness window) are in src/community/photos.py and share this database.

Guarantees (each is covered by tests/test_community.py)
  * A report counts as saved only after the row has been committed AND read back from the database; otherwise a ReportError is raised.
  * Input is cleaned and validated (length, control characters, known category); all SQL is parameterised; text is stored as plain
    text and must be displayed as plain text.
  * Safeguards for an anonymous service: a repeated submission token is idempotent (a double click cannot create two rows), an
    identical report within 10 minutes is refused, each browser session may file at most 5 reports per 10 minutes, the whole
    service at most 60 per hour and 10,000 in total. There is no edit or delete operation anywhere in the application.
  * Viewing the page never creates the database; the first saved report does.
"""
import hashlib
import os
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DB_ENV = "COMMUNITY_DB_PATH"
DEFAULT_DB = ROOT / "data" / "community" / "community.db"
SCHEMA_VERSION = 2   # 1 = text reports; 2 = adds the photo_reports table (src/community/photos.py). Migrated only when something is WRITTEN, never by viewing.

CATEGORIES = ("Heavy traffic", "Accident", "Road closure", "Roadwork", "Flooding or weather", "Signal fault", "Other")
MIN_PLACE, MAX_PLACE = 3, 120
MIN_DESC, MAX_DESC = 10, 500
DUPLICATE_WINDOW_S = 600
SESSION_LIMIT = (5, 600)        # reports per session, per seconds
GLOBAL_LIMIT = (60, 3600)
MAX_REPORTS = 10_000
OUTDATED_AFTER_S = 24 * 3600    # shown as "possibly outdated" after a day
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ReportError(Exception):
    """A report problem. `kind` is machine-readable; `message` is safe and actionable for ordinary users (no paths, no stack traces)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


@dataclass(frozen=True)
class Report:
    id: int
    created_at: float          # epoch seconds, UTC, set by the server when the report was saved
    category: str
    place: str
    description: str
    status: str = "unverified"

    def age_s(self, now=None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.created_at)

    def outdated(self, now=None) -> bool:
        return self.age_s(now) > OUTDATED_AFTER_S


def db_path(path=None) -> Path:
    return Path(path) if path else Path(os.environ.get(DB_ENV) or DEFAULT_DB)


# ------------------------------------------------------------------------------------------------ validation
def _clean(text, label, lo, hi, multiline=False) -> str:
    if not isinstance(text, str):
        raise ReportError("invalid", f"{label} is required.")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if _CONTROL.search(text):
        raise ReportError("invalid", f"{label} contains characters that are not allowed. Please type plain text.")
    text = re.sub(r"\n{3,}", "\n\n", text).strip() if multiline else " ".join(text.split())
    if len(text) < lo:
        raise ReportError("invalid", f"{label} is too short: write at least {lo} characters.")
    if len(text) > hi:
        raise ReportError("invalid", f"{label} is too long: {len(text)} characters, the limit is {hi}. Please shorten it.")
    return text


def validate(category, place, description):
    """Returns the cleaned (category, place, description) or raises ReportError(kind='invalid')."""
    if category not in CATEGORIES:
        raise ReportError("invalid", "Choose a category from the list.")
    return (category, _clean(place, "The road or place", MIN_PLACE, MAX_PLACE),
            _clean(description, "The description", MIN_DESC, MAX_DESC, multiline=True))


def _fingerprint(category, place, description) -> str:
    return hashlib.sha256("|".join((category, place.lower(), " ".join(description.lower().split()))).encode("utf-8")).hexdigest()


def _pseudonym(session_id) -> str:
    return hashlib.sha256(("session:" + str(session_id)).encode("utf-8")).hexdigest()[:16]  # not reversible; no personal data is kept


# ------------------------------------------------------------------------------------------------ storage
def _storage_error(e) -> ReportError:
    return ReportError("storage", "The report could not be stored because the report database is unavailable "
                                  f"({type(e).__name__}). Check that the folder data/community can be written, then try again.")


def _connect(path: Path, create: bool):
    if not path.is_file() and not create:
        return None
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5, isolation_level=None)  # explicit transactions below
    con.row_factory = sqlite3.Row
    try:
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            con.close()
            raise ReportError("storage", "The report database was created by a newer version of this application and cannot be used.")
        if version < 1:
            con.executescript("""
                BEGIN;
                CREATE TABLE IF NOT EXISTS reports (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  REAL    NOT NULL,
                    category    TEXT    NOT NULL,
                    place       TEXT    NOT NULL,
                    description TEXT    NOT NULL,
                    status      TEXT    NOT NULL DEFAULT 'unverified',
                    token       TEXT    UNIQUE,
                    fingerprint TEXT    NOT NULL,
                    submitter   TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reports_created ON reports (created_at);
                CREATE INDEX IF NOT EXISTS idx_reports_submitter ON reports (submitter, created_at);
                PRAGMA user_version = 1;
                COMMIT;""")
            version = 1
        if create and version < 2:   # migration 1 -> 2: only a write may change the file; viewing an older database leaves it exactly as it is
            con.executescript("""
                BEGIN;
                CREATE TABLE IF NOT EXISTS photo_reports (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id      TEXT    NOT NULL UNIQUE,
                    created_at     REAL    NOT NULL,
                    created_at_utc TEXT    NOT NULL,
                    expires_at     REAL    NOT NULL,
                    latitude       REAL    NOT NULL,
                    longitude      REAL    NOT NULL,
                    accuracy_m     REAL,
                    description    TEXT    NOT NULL DEFAULT '',
                    image_file     TEXT,
                    image_sha256   TEXT    NOT NULL,
                    status         TEXT    NOT NULL DEFAULT 'unverified',
                    token          TEXT    UNIQUE,
                    submitter      TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_photo_created ON photo_reports (created_at);
                CREATE INDEX IF NOT EXISTS idx_photo_position ON photo_reports (latitude, longitude);
                CREATE INDEX IF NOT EXISTS idx_photo_submitter ON photo_reports (submitter, created_at);
                PRAGMA user_version = 2;
                COMMIT;""")
    except BaseException:  # never leave a file handle open (a corrupt file raises here)
        con.close()
        raise
    return con


def _row(r) -> Report:
    return Report(r["id"], r["created_at"], r["category"], r["place"], r["description"], r["status"])


def submit(category, place, description, *, session_id, token=None, now=None, path=None):
    """
    Validate and store one report. Returns (Report, outcome) where outcome is "saved" or "already_saved" (the same token was used before).
    Raises ReportError otherwise; nothing is stored in that case. The Report returned has been read back from the database.
    """
    category, place, description = validate(category, place, description)
    now = time.time() if now is None else now
    p, fp, who = db_path(path), None, _pseudonym(session_id)
    fp = _fingerprint(category, place, description)
    try:
        with closing(_connect(p, create=True)) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                if token:
                    old = con.execute("SELECT * FROM reports WHERE token = ?", (token,)).fetchone()
                    if old:
                        con.execute("COMMIT")
                        return _row(old), "already_saved"
                if con.execute("SELECT COUNT(*) FROM reports").fetchone()[0] >= MAX_REPORTS:
                    raise ReportError("full", "The report store is full, so no new reports are accepted. Please tell the person who runs this dashboard.")
                n_session = con.execute("SELECT COUNT(*) FROM reports WHERE submitter = ? AND created_at > ?", (who, now - SESSION_LIMIT[1])).fetchone()[0]
                if n_session >= SESSION_LIMIT[0]:
                    raise ReportError("rate_limited", f"You have already sent {SESSION_LIMIT[0]} reports in the last {SESSION_LIMIT[1] // 60} minutes. Please wait a little before sending another.")
                if con.execute("SELECT COUNT(*) FROM reports WHERE created_at > ?", (now - GLOBAL_LIMIT[1],)).fetchone()[0] >= GLOBAL_LIMIT[0]:
                    raise ReportError("rate_limited", "Many reports were sent in the last hour. Please try again later.")
                dup = con.execute("SELECT created_at FROM reports WHERE fingerprint = ? AND created_at > ? ORDER BY created_at DESC LIMIT 1",
                                  (fp, now - DUPLICATE_WINDOW_S)).fetchone()
                if dup:
                    raise ReportError("duplicate", f"An identical report was already sent {max(1, int((now - dup[0]) // 60))} minute(s) ago, so it was not saved again.")
                cur = con.execute("INSERT INTO reports (created_at, category, place, description, token, fingerprint, submitter) VALUES (?,?,?,?,?,?,?)",
                                  (now, category, place, description, token, fp, who))
                new_id = cur.lastrowid
                con.execute("COMMIT")
            except BaseException:
                if con.in_transaction:
                    con.execute("ROLLBACK")
                raise
        chk = _connect(p, create=False)  # read it back through a NEW connection: only then is it saved
        row = None
        if chk is not None:
            with closing(chk):
                row = chk.execute("SELECT * FROM reports WHERE id = ?", (new_id,)).fetchone()
        if row is None or row["description"] != description:
            raise ReportError("not_saved", "The report could not be confirmed in the database, so it was not saved. Please try again.")
        return _row(row), "saved"
    except ReportError:
        raise
    except sqlite3.Error as e:
        raise _storage_error(e) from None
    except OSError as e:
        raise _storage_error(e) from None


def recent(limit=50, *, path=None):
    """Newest first. An absent database means no reports (it is not created by reading)."""
    try:
        con = _connect(db_path(path), create=False)
        if con is None:
            return []
        with closing(con):
            return [_row(r) for r in con.execute("SELECT * FROM reports ORDER BY created_at DESC, id DESC LIMIT ?", (int(limit),))]
    except ReportError:
        raise
    except (sqlite3.Error, OSError) as e:
        raise _storage_error(e) from None


def count(*, path=None) -> int:
    try:
        con = _connect(db_path(path), create=False)
        if con is None:
            return 0
        with closing(con):
            return con.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    except ReportError:
        raise
    except (sqlite3.Error, OSError) as e:
        raise _storage_error(e) from None


def health(*, path=None) -> dict:
    """Real state of the store, for the status panels: {'state': ok|warn|bad, 'summary', 'count'}."""
    try:
        n = count(path=path)
    except ReportError as e:
        return {"state": "bad", "summary": e.message, "count": None}
    if not db_path(path).is_file():
        return {"state": "ok", "summary": "ready; no reports yet (the database is created when the first report is saved)", "count": 0}
    return {"state": "ok", "summary": f"working; {n} report(s) stored", "count": n}
