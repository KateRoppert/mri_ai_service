"""
Pytest configuration for backend tests.

Route the database at an isolated temp file for the whole backend test session,
set BEFORE any test module imports `database` (which binds SessionLocal/engine at
import time from settings.database_url). Without this, tests that use SessionLocal
either hit the real brain_lesion.db or the container-only /app path, and their
outcome depends on import order across the suite.

setdefault respects an explicit override (e.g. a CI-provided DATABASE_URL).
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "DATABASE_URL",
    f"sqlite:///{Path(tempfile.mkdtemp()) / 'test_backend.db'}",
)

# Create the full schema in the temp DB so DB-backed tests find their tables
# regardless of collection order. Imported only after DATABASE_URL is set, so the
# engine binds to the temp file. Both metadata sets share the same engine.
from database import init_db  # noqa: E402
from patient_registry import ensure_tables  # noqa: E402

init_db()
ensure_tables()
