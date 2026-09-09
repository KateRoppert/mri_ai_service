import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCH_DIR.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (BENCH_DIR, SCRIPTS_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
