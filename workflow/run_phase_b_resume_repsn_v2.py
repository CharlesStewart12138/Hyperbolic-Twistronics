from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from representation.wedderburn_resumable_repsn_v2 import install

install()

from run_phase_b_resume import main


if __name__ == "__main__":
    raise SystemExit(main())
