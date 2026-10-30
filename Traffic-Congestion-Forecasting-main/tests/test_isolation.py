"""The test run must never touch the user's real community database (data/community/community.db)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _isolation  # noqa: E402


class CommunityDatabaseIsolationTests(unittest.TestCase):
    def test_the_environment_points_at_a_temporary_database_outside_the_project(self):
        path = Path(os.environ["COMMUNITY_DB_PATH"]).resolve()
        self.assertEqual(path, _isolation.TEST_DB.resolve())
        self.assertNotEqual(path, _isolation.REAL_DB.resolve())
        self.assertNotIn(_isolation.ROOT.resolve(), path.parents)

    def test_the_store_resolves_to_the_temporary_path_not_the_real_one(self):
        from src.community import store
        self.assertEqual(store.db_path().resolve(), _isolation.TEST_DB.resolve())
        self.assertNotEqual(store.db_path().resolve(), store.DEFAULT_DB.resolve())

    def test_the_real_database_is_unchanged_since_this_process_started(self):
        # size, modification time and content hash: any test that wrote, touched, reset or deleted it would change one of them
        self.assertEqual(_isolation.fingerprint(), _isolation.REAL_DB_AT_START)


if __name__ == "__main__":
    unittest.main()
