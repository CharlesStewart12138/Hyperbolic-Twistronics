from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_final_figures_v2 import install  # noqa: E402

install()

from finalize_extension import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
