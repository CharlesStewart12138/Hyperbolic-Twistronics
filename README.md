# Hyperbolic Twistronics Computer-Assisted Validation

## Repository layout

```text
.
├── .github/workflows/                 GitHub Actions checks
├── certificates/                      final compact certificates
├── configs/                           frozen validation configurations
├── data/figure_sources/               minimal frozen inputs used
├── deep_resolution/                   second research extension, without historical runs
├── environment/                       environment provenance
├── figures/{png,svg,pdf}/             current publication figures only
├── manifests/figure_release           figure hashes, source registry, and QA records
├── postvalidation_resolution/         first research extension, without historical runs
├── public_data/                       public external baselines
├── references/                        immutable manuscript/reference inputs
├── reports/                           final reports and release documentation
├── src/                               scientific implementation
├── tests/                             regression and contract tests
├── tools/                             release and figure utilities
└── workflow/                          executable validation workflows
```

See `docs/REPOSITORY_LAYOUT.md` for the retention policy and provenance details.

## Install

Python 3.12 is required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

Some representation-theory workflows additionally require GAP/Repsn, and some exact group workflows require KBMAG or SageMath. Their absence must be reported as an implementation/environment limitation, never as a scientific failure.

## Test

```bash
python -m pytest -q
```

## Re-render the current publication figures

The plotting entry point reads only frozen scientific data and does not compute scientific results:

```bash
python tools/render_publication_figures.py \
  --project-root . \
  --output-root build/publication_figures
```

The source registry is `manifests/figure_release/valid_figure_registry.json`. Every retained source carries a SHA-256 or directory-tree digest.
