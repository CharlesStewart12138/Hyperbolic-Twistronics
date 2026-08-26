# Post-validation targeted resolution

This directory is an isolated research extension of the frozen validation project. It may read frozen predecessor artifacts by verified hash, but it never edits the original task manifest, adjudications, runs, certificates, reports, workbooks, publication data, or `FINAL_VALIDATION_STATUS.json`.

Execution order is R8, R9, R10, then R16. New estimators, towers, cutoffs, sectors, normalizations, metrics, and acceptance rules are frozen in `configs/` before their outcomes are inspected. Every extension task terminates with one of the six statuses authorized by the post-completion instruction.
