"""
Community traffic PHOTO reports: a photograph taken at the reporter's current location, shown to other people who look at the same area
while it is still fresh. Stored on THIS computer in the same SQLite database as the text reports (src/community/store.py) and in
data/community/photos/ (both git-ignored; override the database with COMMUNITY_DB_PATH, the photo folder is always next to it).

What is and is not claimed
  * The location is what the reporter's BROWSER said (navigator.geolocation). The server cannot prove that it is the true position of the
    phone: a determined person could forge a request. The report is therefore always shown as "unverified" and as "location reported by the
    device". There is no way to type coordinates in the application.
  * A photo is "fresh" for COMMUNITY_PHOTO_FRESH_MINUTES (default 10) after it was UPLOADED. The upload time is the SERVER's UTC clock, never the
    reporter's device clock, and freshness is always computed from it. After that the report is "expired": it is never offered as current traffic.
  * Photos are community evidence of what someone saw. They are not traffic measurements, are not mixed into the TomTom data and are never
    used by the forecasting models.

Safety (each point is covered by tests/test_photo_reports.py)
  * The upload is decoded and RE-ENCODED by Pillow: only JPEG, PNG and WebP are accepted (judged by the file's own content, not its name or type),
    animated images, huge images (> 40 megapixels) and files over COMMUNITY_PHOTO_MAX_MB (default 8) are refused, and the stored JPEG carries no
    metadata at all (no EXIF GPS, camera model or time), so what the reporter's phone wrote into the file is not published.
  * File names are random (uuid4); no name from the user is ever used for a path. Photos are only ever shown as images, never executed.
  * Coordinates must be real numbers inside the map's area (configurable); the GPS accuracy must be reported and good enough.
  * Rate limits per browser session and for the whole service; an identical photo within the freshness window is refused. The reporter is
    stored only as a one-way pseudonym of the session id and is never shown to anyone.
  * Old photo FILES are deleted after COMMUNITY_PHOTO_RETENTION_HOURS (default 24); the row (without the image) stays as history and does not count
    towards the limit of 2,000 stored photos.
"""
import base64
import binascii
import datetime as dt
import hashlib
import io
import math
import os
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from src.community import store
from src.live import geo

# ------------------------------------------------------------------------------------------------ configuration (environment, with safe limits)
ENV_FRESH_MIN = "COMMUNITY_PHOTO_FRESH_MINUTES"
ENV_RADIUS_KM = "COMMUNITY_PHOTO_RADIUS_KM"
ENV_MAX_MB = "COMMUNITY_PHOTO_MAX_MB"
ENV_MAX_ACCURACY_M = "COMMUNITY_PHOTO_MAX_ACCURACY_M"
ENV_RETENTION_H = "COMMUNITY_PHOTO_RETENTION_HOURS"
ENV_REFRESH_S = "COMMUNITY_PHOTO_REFRESH_S"
ENV_SCOPE = "COMMUNITY_PHOTO_SCOPE"          # "bengaluru" (default: the area this map covers) or "any"


def _env_number(name, default, lo, hi):
    """A number from the environment, clamped to a safe range; anything unusable falls back to the default."""
    try:
        v = float(os.environ.get(name, "").strip())
    except ValueError:
        return float(default)
    return min(max(v, lo), hi) if math.isfinite(v) else float(default)


def fresh_seconds() -> float:
    return _env_number(ENV_FRESH_MIN, 10, 0.5, 24 * 60) * 60


def radius_km() -> float:
    return _env_number(ENV_RADIUS_KM, 2.0, 0.1, 25.0)


def max_upload_bytes() -> int:
    return int(_env_number(ENV_MAX_MB, 8, 0.1, 25) * 1024 * 1024)


def max_accuracy_m() -> float:
    return _env_number(ENV_MAX_ACCURACY_M, 1000, 10, 50_000)


def retention_seconds() -> float:
    return _env_number(ENV_RETENTION_H, 24, 0.1, 24 * 30) * 3600


def refresh_seconds() -> int:
    return int(_env_number(ENV_REFRESH_S, 30, 5, 600))


ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
MAX_PIXELS = 40_000_000
MIN_SIDE = 64
MAX_SIDE, THUMB_SIDE = 1600, 360
DESC_MAX = 200
SESSION_LIMIT = (3, 600)         # photos per browser session, per seconds
GLOBAL_LIMIT = (30, 600)
MAX_PHOTOS = 2000                # photo FILES stored at once; reports whose file was deleted by the retention rule (history) do not count
NEARBY_LIMIT = 40
PROVENANCE = "Location reported by the reporter's device (browser GPS); not verified by the server."


class PhotoError(Exception):
    """A photo-report problem. `kind` is machine-readable; `message` is safe and actionable (no paths, no stack traces)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


@dataclass(frozen=True)
class PhotoReport:
    report_id: str
    created_at: float            # epoch seconds, UTC, set by the SERVER when the photo was saved
    expires_at: float            # created_at + the freshness window in force at that time
    latitude: float
    longitude: float
    accuracy_m: float
    description: str
    image_file: str              # "" once the file has been deleted by the retention rule
    status: str = "unverified"

    @property
    def created_utc(self) -> str:
        return dt.datetime.fromtimestamp(self.created_at, dt.timezone.utc).isoformat(timespec="seconds")

    def age_s(self, now=None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.created_at)

    def is_fresh(self, now=None) -> bool:
        return (time.time() if now is None else now) < self.expires_at

    def state(self, now=None) -> str:
        return "fresh" if self.is_fresh(now) else "expired"


@dataclass(frozen=True)
class NearbyPhoto:
    report: PhotoReport
    distance_km: float


# ------------------------------------------------------------------------------------------------ wording (shared by the pages and the tests)
def uploaded_text(report: PhotoReport, now=None) -> str:
    age = report.age_s(now)
    if age < 60:
        return "Uploaded just now"
    minutes = int(age // 60)
    if minutes < 60:
        return f"Uploaded {minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = int(minutes // 60)
    return f"Uploaded {hours} hour{'s' if hours != 1 else ''} ago" if hours < 48 else f"Uploaded {hours // 24} days ago"


def distance_text(km: float) -> str:
    return "less than 0.1 km from your searched location" if km < 0.1 else f"{km:.1f} km from your searched location"


def state_text(report: PhotoReport, now=None) -> str:
    if report.is_fresh(now):
        left = max(0, int((report.expires_at - (time.time() if now is None else now)) // 60) + 1)
        return f"Fresh: shown as current for about {left} more minute{'s' if left != 1 else ''}"
    return "Expired: not current traffic"


# ------------------------------------------------------------------------------------------------ validation
def _clean_description(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise PhotoError("invalid", "The description must be text.")
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if any(ord(c) < 32 or ord(c) == 127 for c in text):
        raise PhotoError("invalid", "The description contains characters that are not allowed. Please type plain text.")
    if len(text) > DESC_MAX:
        raise PhotoError("invalid", f"The description is too long: {len(text)} characters, the limit is {DESC_MAX}.")
    return text


def validate_location(lat, lon, accuracy_m):
    """Server-side check of what the browser reported. Returns (lat, lon, accuracy_m) or raises PhotoError('location', ...)."""
    if lat is None or lon is None:
        raise PhotoError("location", "No GPS position was received. A photo report needs the position of your device: allow location access and try again.")
    try:
        bbox = None if os.environ.get(ENV_SCOPE, "").strip().lower() == "any" else geo.BENGALURU_BBOX
        lat, lon = geo.validate_location(lat, lon, bbox=bbox)
    except geo.LocationError as e:
        raise PhotoError("location", f"The reported position cannot be used. {e}") from None
    if accuracy_m is None or isinstance(accuracy_m, bool) or not isinstance(accuracy_m, (int, float)) or not math.isfinite(accuracy_m) or accuracy_m < 0:
        raise PhotoError("location", "The device did not report how accurate its position is, so the position cannot be used.")
    if accuracy_m > max_accuracy_m():
        raise PhotoError("location", f"Your device's position is too imprecise (about {accuracy_m:,.0f} m; at most {max_accuracy_m():,.0f} m is accepted). "
                                     "Move to a place with better GPS reception and try again.")
    return lat, lon, float(accuracy_m)


def decode_data_url(value) -> bytes:
    """The component sends the photo as a data URL; returns the raw bytes or raises PhotoError('invalid_image')."""
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value[:64]:
        raise PhotoError("invalid_image", "No photo was received. Take a photo or choose one, then try again.")
    payload = value.split(";base64,", 1)[1]
    if len(payload) > (max_upload_bytes() * 4) // 3 + 8:
        raise PhotoError("too_large", f"The photo is too large (the limit is {max_upload_bytes() // (1024 * 1024)} MB). Try a smaller photo.")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise PhotoError("invalid_image", "The photo could not be read. Take it again or choose a different one.") from None


def process_image(data: bytes):
    """Validate an upload by its CONTENT and return (main_jpeg, thumbnail_jpeg, sha256_of_main). Raises PhotoError. Nothing is written here."""
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise PhotoError("invalid_image", "No photo was received. Take a photo or choose one, then try again.")
    if len(data) > max_upload_bytes():
        raise PhotoError("too_large", f"The photo is too large ({len(data) / 1048576:.1f} MB; the limit is {max_upload_bytes() // (1024 * 1024)} MB). Try a smaller photo.")
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(bytes(data))) as im:
            if im.format not in ALLOWED_FORMATS:
                raise PhotoError("invalid_format", "Only JPEG, PNG or WebP photos are accepted.")
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise PhotoError("too_large", "The photo has too many pixels. Try a smaller photo.")
            if min(w, h) < MIN_SIDE:
                raise PhotoError("invalid_image", f"The photo is too small (at least {MIN_SIDE} pixels on each side).")
            if getattr(im, "n_frames", 1) > 1:
                raise PhotoError("invalid_format", "Animated images are not accepted. Use a single photo.")
            im.load()                                     # a truncated or corrupt file fails HERE, before anything is stored
            im = ImageOps.exif_transpose(im)               # apply the camera's rotation, then the metadata is dropped below
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.getchannel("A"))
                im = flat
            else:
                im = im.convert("RGB")
            main = im.copy()
            main.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            thumb = im.copy()
            thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
    except PhotoError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, MemoryError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise PhotoError("invalid_image", "The file is not a readable photo (JPEG, PNG or WebP). Take the photo again or choose a different one.") from None
    out = []
    for img, quality in ((main, 82), (thumb, 70)):
        clean = Image.frombytes("RGB", img.size, img.tobytes())   # a brand-new image: only pixels. A JPEG comment, PNG text chunks, XMP or an ICC profile
        buf = io.BytesIO()                                       # in the upload live in img.info and would otherwise be copied into the saved file
        clean.save(buf, "JPEG", quality=quality, optimize=True)
        out.append(buf.getvalue())
    return out[0], out[1], hashlib.sha256(out[0]).hexdigest()


# ------------------------------------------------------------------------------------------------ storage
def photos_dir(path=None) -> Path:
    return store.db_path(path).parent / "photos"


def _row(r) -> PhotoReport:
    return PhotoReport(r["report_id"], r["created_at"], r["expires_at"], r["latitude"], r["longitude"], r["accuracy_m"] or 0.0,
                       r["description"], r["image_file"] or "", r["status"])


def _has_table(con) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'photo_reports'").fetchone() is not None


def _write_atomic(target: Path, data: bytes):
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def submit_photo(image, lat, lon, description="", *, session_id, accuracy_m=None, token=None, now=None, path=None):
    """
    Validate and store one photo report. `image` is the raw bytes (or a data URL). Returns (PhotoReport, outcome), outcome "saved" or
    "already_saved" (the same token was used before). Raises PhotoError; nothing is stored in that case (files are removed again).
    The report returned was read back from the database, and its file was checked to exist.
    """
    data = decode_data_url(image) if isinstance(image, str) else image
    lat, lon, acc = validate_location(lat, lon, accuracy_m)
    desc = _clean_description(description)
    main_jpeg, thumb_jpeg, digest = process_image(data)
    now = time.time() if now is None else now           # the SERVER's clock: freshness never depends on the reporter's device
    p, who = store.db_path(path), store._pseudonym(session_id)
    folder = photos_dir(path)
    report_id = uuid.uuid4().hex
    names = (f"{report_id}.jpg", f"{report_id}_t.jpg")
    written = []
    try:
        with closing(store._connect(p, create=True)) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                if token:
                    old = con.execute("SELECT * FROM photo_reports WHERE token = ?", (token,)).fetchone()
                    if old:
                        con.execute("COMMIT")
                        return _row(old), "already_saved"
                if con.execute("SELECT COUNT(*) FROM photo_reports WHERE image_file IS NOT NULL AND image_file != ''").fetchone()[0] >= MAX_PHOTOS:
                    raise PhotoError("full", "The photo store is full, so no new photos are accepted. Please tell the person who runs this dashboard.")
                if con.execute("SELECT COUNT(*) FROM photo_reports WHERE submitter = ? AND created_at > ?", (who, now - SESSION_LIMIT[1])).fetchone()[0] >= SESSION_LIMIT[0]:
                    raise PhotoError("rate_limited", f"You have already sent {SESSION_LIMIT[0]} photos in the last {SESSION_LIMIT[1] // 60} minutes. Please wait a little.")
                if con.execute("SELECT COUNT(*) FROM photo_reports WHERE created_at > ?", (now - GLOBAL_LIMIT[1],)).fetchone()[0] >= GLOBAL_LIMIT[0]:
                    raise PhotoError("rate_limited", "Many photos were sent in the last few minutes. Please try again shortly.")
                dup = con.execute("SELECT created_at FROM photo_reports WHERE image_sha256 = ? AND expires_at > ? LIMIT 1", (digest, now)).fetchone()
                if dup:
                    raise PhotoError("duplicate", "This exact photo was already sent a moment ago, so it was not saved again.")
                folder.mkdir(parents=True, exist_ok=True)
                for name, blob in zip(names, (main_jpeg, thumb_jpeg)):
                    _write_atomic(folder / name, blob)
                    written.append(folder / name)
                created_utc = dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(timespec="seconds")
                con.execute("INSERT INTO photo_reports (report_id, created_at, created_at_utc, expires_at, latitude, longitude, accuracy_m, description, "
                            "image_file, image_sha256, status, token, submitter) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (report_id, now, created_utc, now + fresh_seconds(), lat, lon, acc, desc, names[0], digest, "unverified", token, who))
                con.execute("COMMIT")
            except BaseException:
                if con.in_transaction:
                    con.execute("ROLLBACK")
                raise
        row = None
        chk = store._connect(p, create=False)               # read it back through a NEW connection: only then is it saved
        if chk is not None:
            with closing(chk):
                row = chk.execute("SELECT * FROM photo_reports WHERE report_id = ?", (report_id,)).fetchone()
        if row is None or not (folder / names[0]).is_file():
            raise PhotoError("not_saved", "The photo could not be confirmed in storage, so it was not saved. Please try again.")
        return _row(row), "saved"
    except PhotoError:
        _discard(written)
        raise
    except store.ReportError as e:
        _discard(written)
        raise PhotoError("storage", e.message) from None
    except (sqlite3.Error, OSError) as e:
        _discard(written)
        raise PhotoError("storage", "The photo could not be stored because the storage is unavailable "
                                    f"({type(e).__name__}). Check that the folder data/community can be written, then try again.") from None


def _discard(files):
    for f in files:
        try:
            Path(f).unlink()
        except OSError:
            pass


def _query(sql, args, path):
    """Rows (newest first) or [] when there is no database or no photo table yet. Reading never creates or migrates anything."""
    try:
        con = store._connect(store.db_path(path), create=False)
        if con is None:
            return []
        with closing(con):
            if not _has_table(con):
                return []
            return [_row(r) for r in con.execute(sql, args)]
    except store.ReportError:
        raise
    except (sqlite3.Error, OSError) as e:
        raise PhotoError("storage", f"The photo reports could not be read ({type(e).__name__}).") from None


def nearby(lat, lon, radius=None, *, now=None, include_expired=False, limit=NEARBY_LIMIT, path=None):
    """
    Photo reports within `radius` km (default: COMMUNITY_PHOTO_RADIUS_KM, 2 km) of the point, nearest-in-time first. By default ONLY fresh ones:
    an expired report is never returned as current traffic. Distance is the great-circle distance, never a place-name match, and a report
    exactly beyond the radius is never included.
    """
    lat, lon = float(lat), float(lon)
    radius = radius_km() if radius is None else float(radius)
    now = time.time() if now is None else now
    dlat = radius / 111.32
    dlon = radius / (111.32 * max(0.05, math.cos(math.radians(lat))))
    sql = ("SELECT * FROM photo_reports WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
           + ("" if include_expired else " AND expires_at > ?") + " ORDER BY created_at DESC, id DESC LIMIT 1000")
    args = (lat - dlat, lat + dlat, lon - dlon, lon + dlon) + (() if include_expired else (now,))
    out = []
    for r in _query(sql, args, path):
        d = geo.distance_km(lat, lon, r.latitude, r.longitude)
        if d <= radius:
            out.append(NearbyPhoto(r, d))
    return out[:int(limit)]


def history(limit=50, *, path=None):
    """Every photo report newest first, fresh and expired (the report history); expired ones are labelled as such by state()."""
    return _query("SELECT * FROM photo_reports ORDER BY created_at DESC, id DESC LIMIT ?", (int(limit),), path)


def count_fresh(*, now=None, path=None) -> int:
    now = time.time() if now is None else now
    return len(_query("SELECT * FROM photo_reports WHERE expires_at > ?", (now,), path))


def read_image(report: PhotoReport, *, thumb=False, path=None):
    """The stored JPEG bytes of a report (or None once the retention rule has deleted the file). The name comes from the database and is checked."""
    name = report.image_file
    if not name or "/" in name or "\\" in name or ".." in name or not name.endswith(".jpg"):
        return None
    if thumb:
        name = name[:-4] + "_t.jpg"
    f = photos_dir(path) / name
    try:
        return f.read_bytes() if f.is_file() else None
    except OSError:
        return None


def thumbnail_data_url(report: PhotoReport, *, path=None):
    data = read_image(report, thumb=True, path=path)
    return None if data is None else "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def purge(*, now=None, path=None) -> int:
    """Delete the photo FILES of reports older than the retention time (default 24 h); the rows stay as history without an image. Returns the count."""
    now = time.time() if now is None else now
    old = _query("SELECT * FROM photo_reports WHERE created_at < ? AND image_file IS NOT NULL AND image_file != ''", (now - retention_seconds(),), path)
    if not old:
        return 0
    n = 0
    try:
        with closing(store._connect(store.db_path(path), create=False)) as con:
            for r in old:
                _discard([photos_dir(path) / r.image_file, photos_dir(path) / (r.image_file[:-4] + "_t.jpg")])
                con.execute("UPDATE photo_reports SET image_file = NULL, status = 'expired' WHERE report_id = ?", (r.report_id,))
                n += 1
    except (sqlite3.Error, OSError, store.ReportError):
        return n
    return n


# ------------------------------------------------------------------------------------------------ what the capture component reports
CAPTURE_MESSAGES = {
    "denied": "Location permission was denied. A photo report needs the GPS position of your device so that other people can see where the "
              "traffic is. Allow location access for this site in your browser settings, then press \"Share my location\" again.",
    "insecure": "Your browser only gives out the GPS position on a secure page (https:// or localhost). Open the dashboard that way, or use it on "
                "the computer that runs it.",
    "unavailable": "Your device could not determine its position. Check that location is switched on and try again outdoors.",
    "timeout": "Finding your position took too long. Check that location is switched on, then try again.",
    "unsupported": "This browser cannot give out a GPS position, so a photo report cannot be sent from it.",
    "camera_denied": "Camera permission was denied. You can choose an existing photo instead, or allow camera access for this site in your browser settings.",
    "camera_unavailable": "No usable camera was found. You can choose an existing photo instead.",
    "unsupported_image": "This photo cannot be used. Take it again or choose a JPEG, PNG or WebP photo.",
    "image_too_large": "The photo is too large. Try a smaller photo.",
}


def explain_capture(payload) -> dict:
    """
    Turn the component's value into {'ready': bool, 'image': data URL|None, 'lat', 'lon', 'accuracy_m', 'messages': [..]}. Pure and safe on any input:
    a missing or malformed payload is simply 'not ready'. `ready` means an image AND a reported position are present; the server checks them again.
    """
    p = payload if isinstance(payload, dict) else {}
    messages = []
    for key in (p.get("loc_status"), p.get("cam_status"), p.get("image_error")):   # distinct codes: the location and camera ones never collide
        text = CAPTURE_MESSAGES.get(key) if isinstance(key, str) else None
        if text and text not in messages:
            messages.append(text)
    image = p.get("image") if isinstance(p.get("image"), str) and p.get("image").startswith("data:image/") else None
    lat, lon, acc = p.get("lat"), p.get("lon"), p.get("accuracy_m")
    has_pos = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (lat, lon, acc))
    return {"ready": bool(image and has_pos), "image": image, "lat": lat if has_pos else None, "lon": lon if has_pos else None,
            "accuracy_m": acc if has_pos else None, "messages": messages}
