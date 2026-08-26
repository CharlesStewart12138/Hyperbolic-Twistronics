from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

import run_r10_dos_v2 as parser_fixed  # noqa: E402
from r10_dos_v2 import slq_local_fixed_memmap  # noqa: E402


parser_fixed.implementation.slq_local = slq_local_fixed_memmap


if __name__ == "__main__":
    raise SystemExit(parser_fixed.implementation.main())

