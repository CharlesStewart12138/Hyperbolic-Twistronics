from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXTENSION_ROOT / "workflow"))

import run_r8_spectral as workflow  # noqa: E402
from projected_spectral_v2 import stochastic_chebyshev_moments  # noqa: E402


workflow.stochastic_chebyshev_moments = stochastic_chebyshev_moments


if __name__ == "__main__":
    raise SystemExit(workflow.main())
