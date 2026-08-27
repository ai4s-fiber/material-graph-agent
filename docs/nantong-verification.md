# Nantong Knowledge-Graph-Only Verification

- Date: `2026-08-27` (Asia/Shanghai)
- Host: `user-SYS-740GP-TNRT`
- User: `cyy`
- Checkout: `/home/cyy/dhu-zh-workspace/omnimat/worktrees/material-knowledge-graph-public`
- Source snapshot: `6236ed195983e6e34543eb98b39f1a82990f0e7d`
- GitHub identity for this repository: `fafasco16` via repository-local SSH configuration

The server default SSH identity remained unchanged. No production container, Compose project, runtime configuration, database, proxy, DNS record, or other server project was modified.

## Scope proof

- 167 files in the final working tree before Git metadata.
- No workflow runtime, FastAPI API, SSE, UI, agent memory, prediction, experiment, task-subgraph, agent-view, or task-evidence implementation.
- Kept only ingestion, parsing, fact extraction, review, PostgreSQL/AGE, LightRAG, provenance, admission, audit, layout, supporting provider coordination, tests, and documentation.

## Verification

- Full exported pytest suite passed; 5 external or dedicated-environment cases skipped as designed.
- Python compile check passed with bytecode redirected to container `/tmp`.
- Ruff `0.12.0` passed with cache disabled.
- Gitleaks `8.30.1` scanned approximately 2.22 MB and found no leaks.
- Tests ran in disposable containers with `--network none`.

## Publication guard

The replacement branch is a new root history. Publication must use the exact lease `refs/heads/main:6236ed195983e6e34543eb98b39f1a82990f0e7d`; a remote change causes the push to fail rather than overwrite unreviewed work.
