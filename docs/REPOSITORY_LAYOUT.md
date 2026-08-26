# Repository layout and retention policy

## Purpose

This directory is the compact GitHub-facing copy of the validation project. It is designed for code review, testing, inspection of final scientific conclusions, and deterministic re-rendering of the current publication figures.

## Retained

- Scientific source under `src/`.
- Executable workflows, tests, and frozen configurations.
- Core, post-validation, and deep-resolution final status files.
- Final reports, theorem/computation matrices, error budgets, and certificates.
- Public reproduction inputs and immutable manuscript/reference inputs.
- The current 18 publication figures in PNG, SVG, and PDF.
- Exactly the frozen result files referenced by the current figure-source registry.
- Hash mappings from the original run-tree paths to compact distribution paths.

## Excluded

- Historical `results/<run-id>/` trees not used by the current figures.
- Previous figure renderings and same-number superseded variants.
- Recovery checkpoints, patch files, probes, smoke outputs, QA contact sheets, and temporary workbooks.
- `.venv`, `node_modules`, bytecode, test caches, and local tool downloads.
- The original `.git` directory; initialize a new repository for this distribution if desired.

## Provenance

`manifests/figure_release/source_relocation.json` maps original result paths to compact paths. File contents are copied without transformation and are verified against the original SHA-256 or tree-inventory digest.

`RELEASE_STATUS.json` records final file counts, sizes, hash-audit results, test status, and Python line-count statistics for the compact copy.

The original project remains unchanged and is the authoritative historical execution archive.
