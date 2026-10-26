"""CLI entry point for the distributed-pipeline dashboard."""

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app import main  # noqa: E402


if __name__ == "__main__":
    main()
