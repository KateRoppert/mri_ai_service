import sys
from pathlib import Path

# Put scripts/ on sys.path so `preprocessing_steps.skull_stripping` imports like Stage 05.
SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
