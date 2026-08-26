from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_r10_dos as implementation  # noqa: E402


_load_yaml = implementation.load_yaml


def load_yaml_with_frozen_parser_fix(path: Path) -> dict[str, object]:
    value = _load_yaml(path)
    if path.name == "r10_preregistration.yaml":
        fix = _load_yaml(EXTENSION_ROOT / "configs" / "r10_acceptance_parser_fix_preregistration.yaml")
        value["acceptance"]["at_least_two_towers_must_pass"] = int(fix["frozen_interpretation"])
    return value


implementation.load_yaml = load_yaml_with_frozen_parser_fix


if __name__ == "__main__":
    raise SystemExit(implementation.main())

