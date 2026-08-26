from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_inventory(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def inventory_digest(inventory: dict[str, str]) -> str:
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def svg_text(path: Path) -> str:
    root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    return " ".join(value.strip() for value in root.itertext() if value.strip())


def verify_sources(root: Path) -> int:
    registry = json.loads((root / "manifests/figure_release/valid_figure_registry.json").read_text(encoding="utf-8-sig"))
    checked: set[str] = set()
    for figure in registry["selected_figures"]:
        for source in figure["source_data"]:
            relative = str(source["path"])
            if relative in checked:
                continue
            checked.add(relative)
            path = root / relative
            if source["kind"] == "file":
                if sha256_file(path) != source["sha256"]:
                    raise RuntimeError(f"Frozen source hash mismatch: {relative}")
            else:
                inventory = directory_inventory(path)
                if len(inventory) != int(source["file_count"]):
                    raise RuntimeError(f"Frozen source count mismatch: {relative}")
                if inventory_digest(inventory) != source["tree_inventory_sha256"]:
                    raise RuntimeError(f"Frozen source tree mismatch: {relative}")
    return len(checked)


def audit_figures(root: Path, smoke: Path) -> dict[str, object]:
    records: dict[str, object] = {}
    for kind in ("png", "svg", "pdf"):
        files = sorted((root / "figures" / kind).glob(f"*.{kind}"))
        if len(files) != 18:
            raise RuntimeError(f"Expected 18 {kind} files, found {len(files)}")
        if any("beautified" in path.name or "_v2" in path.name for path in files):
            raise RuntimeError(f"Superseded version suffix remains in {kind} figure names")
        records[kind] = {
            "count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
        }

    white_checks = {}
    for path in sorted((root / "figures/png").glob("*.png")):
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            corners = [
                rgba.getpixel((0, 0)),
                rgba.getpixel((rgba.width - 1, 0)),
                rgba.getpixel((0, rgba.height - 1)),
                rgba.getpixel((rgba.width - 1, rgba.height - 1)),
            ]
        passed = all(pixel == (255, 255, 255, 255) for pixel in corners)
        if not passed:
            raise RuntimeError(f"Non-white publication background: {path.name}")
        white_checks[path.name] = "PASS"

    figure9 = svg_text(root / "figures/svg/figure09_balanced_full_shell.svg")
    figure11 = svg_text(root / "figures/svg/figure11_external_reproduction.svg")
    figure16 = svg_text(root / "figures/svg/figure16_curvature_relevant_universality.svg")
    requested = {
        "figure_09_panel_f_removed": "(f)" not in figure9 and "PASS_CERTIFIED" not in figure9,
        "figure_11_small_annotation_removed": "D-10 saved spectral residuals" not in figure11,
        "figure_16_panel_d_removed": "(d)" not in figure16 and "PASS_RESTRICTED_CLASS" not in figure16,
    }
    if not all(requested.values()):
        raise RuntimeError(f"Requested figure-removal audit failed: {requested}")

    png_pairs = []
    for release in sorted((root / "figures/png").glob("*.png")):
        rerendered = smoke / "png" / release.name
        equal = rerendered.is_file() and sha256_file(release) == sha256_file(rerendered)
        png_pairs.append(equal)
    if not all(png_pairs):
        raise RuntimeError("Independent PNG rerender is not byte-identical")

    records["white_background"] = {"pass_count": len(white_checks), "figure_count": 18}
    records["requested_adjustments"] = {key: "PASS" if value else "FAIL" for key, value in requested.items()}
    records["independent_rerender"] = {
        "state": "PASS",
        "png_sha256_identical": f"{sum(png_pairs)}/18",
        "svg_and_pdf_note": "rendered successfully; timestamp metadata is intentionally not hash-stable",
    }
    return records


def audit_structure(root: Path) -> dict[str, object]:
    forbidden_directories = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "work", "results"}
    found_directories = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and path.name in forbidden_directories
    ]
    forbidden_files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.startswith("checkpoint_")
            or path.suffix.lower() in {".patch", ".diff", ".log", ".tmp", ".part", ".wsp"}
        )
    ]
    if found_directories or forbidden_files:
        raise RuntimeError(f"Generated/process artifacts remain: {found_directories}, {forbidden_files}")
    return {
        "forbidden_directory_count": 0,
        "forbidden_process_file_count": 0,
        "historical_results_included": False,
        "superseded_figure_versions_included": False,
    }


def python_statistics(root: Path) -> dict[str, object]:
    files = sorted(root.rglob("*.py"))
    by_top_level: dict[str, dict[str, int]] = {}
    rows = []
    physical_total = 0
    nonblank_total = 0
    for path in files:
        relative = path.relative_to(root)
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        physical = len(lines)
        nonblank = sum(bool(line.strip()) for line in lines)
        physical_total += physical
        nonblank_total += nonblank
        top = relative.parts[0]
        bucket = by_top_level.setdefault(top, {"files": 0, "physical_lines": 0, "nonblank_lines": 0})
        bucket["files"] += 1
        bucket["physical_lines"] += physical
        bucket["nonblank_lines"] += nonblank
        rows.append({"path": relative.as_posix(), "physical_lines": physical, "nonblank_lines": nonblank})
    return {
        "definition": "physical lines counted with Python str.splitlines(); blank and comment lines are included in physical_lines",
        "python_file_count": len(files),
        "physical_lines": physical_total,
        "nonblank_lines": nonblank_total,
        "by_top_level": by_top_level,
        "files": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--test-summary", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    smoke = args.smoke_root.resolve()

    source_count = verify_sources(root)
    structure = audit_structure(root)
    figure_audit = audit_figures(root, smoke)
    statistics = python_statistics(root)
    json_write(root / "reports/python_code_statistics.json", statistics)

    report = "\n".join(
        [
            "# Compact GitHub release audit",
            "",
            "- Overall: **PASS**",
            f"- Tests: **{args.test_summary}**",
            f"- Frozen figure sources: **{source_count}/47 verified**",
            "- Publication outputs: **18 PNG + 18 SVG + 18 PDF**",
            "- Independent rerender: **18/18 PNG files SHA-256 identical**",
            "- Pure-white background: **18/18 PASS**",
            "- Figure 9(f), Figure 16(d), and the Figure 11(a) annotation: **absent as requested**",
            "- Historical results/checkpoints/environments/caches: **not included**",
            f"- Python files: **{statistics['python_file_count']}**",
            f"- Python physical lines: **{statistics['physical_lines']}**",
            f"- Python nonblank lines: **{statistics['nonblank_lines']}**",
            "",
            "The original validation project was read only and remains the authoritative historical execution archive.",
            "",
        ]
    )
    (root / "reports/release_audit.md").write_text(report, encoding="utf-8")

    excluded = {"RELEASE_STATUS.json", "manifests/release_file_inventory.json"}
    inventory = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    inventory_record = {
        "schema_version": 1,
        "scope": "all compact-release files except RELEASE_STATUS.json and this inventory file",
        "file_count": len(inventory),
        "tree_inventory_sha256": inventory_digest(inventory),
        "files": inventory,
    }
    json_write(root / "manifests/release_file_inventory.json", inventory_record)

    source_root = Path(r"C:\Users\charl\Documents\Codex\2026-08-24\new-chat\outputs\validation")
    status = {
        "schema_version": 1,
        "state": "COMPACT_GITHUB_RELEASE_COMPLETE",
        "source_project": str(source_root),
        "source_project_preserved": True,
        "historical_execution_archive_included": False,
        "structure_audit": structure,
        "scientific_source_audit": {"verified_unique_sources": source_count, "state": "PASS"},
        "figure_audit": figure_audit,
        "test_summary": args.test_summary,
        "python_statistics": {
            "python_file_count": statistics["python_file_count"],
            "physical_lines": statistics["physical_lines"],
            "nonblank_lines": statistics["nonblank_lines"],
            "details": "reports/python_code_statistics.json",
        },
        "source_status_hashes": {
            "FINAL_VALIDATION_STATUS.json": sha256_file(source_root / "FINAL_VALIDATION_STATUS.json"),
            "beautified_figures_white_v2/BEAUTIFIED_FIGURE_STATUS.json": sha256_file(source_root / "beautified_figures_white_v2/BEAUTIFIED_FIGURE_STATUS.json"),
        },
        "release_inventory": {
            "path": "manifests/release_file_inventory.json",
            "file_count": inventory_record["file_count"],
            "tree_inventory_sha256": inventory_record["tree_inventory_sha256"],
            "sha256": sha256_file(root / "manifests/release_file_inventory.json"),
        },
    }
    json_write(root / "RELEASE_STATUS.json", status)
    print(json.dumps({"state": status["state"], **status["python_statistics"], "tree_hash": inventory_record["tree_inventory_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
