from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPUTE_DIRS = [
    "exact", "geometry", "covers", "model", "spectral", "hodge", "bulk",
    "representation", "dos", "diffraction", "external", "audit",
]


def test_compute_modules_do_not_import_plotting() -> None:
    forbidden = ("import matplotlib", "from matplotlib", "import seaborn", "from seaborn")
    for directory in COMPUTE_DIRS:
        base = ROOT / "src" / directory
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            assert not any(token in text for token in forbidden), path

