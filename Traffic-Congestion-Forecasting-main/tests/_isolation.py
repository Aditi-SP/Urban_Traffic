"""
Test isolation for the community database.

Pages and the router read the community store (the sidebar chip, Community Reports, the home strip). Without an override they open the
user's REAL data/community/community.db. Importing this module points COMMUNITY_DB_PATH at a path inside a temporary folder (removed at exit)
for the whole test process, so no test can read, create, change or delete the real database. Tests that need a populated store still
pass their own temporary path with mock.patch.dict(os.environ, ...); that is restored to THIS temporary path afterwards, never to the real one.

Every test module that runs the app or a page imports this (directly or through _guards). tests/test_isolation.py checks it.
"""
import atexit
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "community" / "community.db"


def fingerprint(path=REAL_DB):
    """(size, mtime_ns, sha256 prefix) of a file, or None when it does not exist. Reads the file; never opens it as a database."""
    try:
        st = path.stat()
        return (st.st_size, st.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()[:16])
    except OSError:
        return None


REAL_DB_AT_START = fingerprint()
_TMP = tempfile.mkdtemp(prefix="tc_test_community_")
TEST_DB = Path(_TMP) / "community.db"   # not created: viewing never creates a database; a test that submits a report creates it here
os.environ["COMMUNITY_DB_PATH"] = str(TEST_DB)
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
