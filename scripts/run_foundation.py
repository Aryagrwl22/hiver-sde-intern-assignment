"""Run from repo root: python scripts/run_foundation.py --demo"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from hiver_foundation.__main__ import main

if __name__ == "__main__":
    main()
