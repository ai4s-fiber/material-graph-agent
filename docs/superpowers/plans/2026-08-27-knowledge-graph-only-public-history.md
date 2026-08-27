# Knowledge-Graph-Only Public Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public repository history with a standalone snapshot containing only material knowledge-graph construction, persistence, validation, and layout code.

**Architecture:** Export an allowlisted tree from the verified public commit on the Nantong server, remove task-agent and workflow modules, and create a new root commit. Force-push with an exact lease so the public branch no longer contains the previously published agent runtime history.

**Tech Stack:** Python 3.11+, Pydantic, PostgreSQL/Apache AGE, LightRAG, NumPy, pytest, Ruff, Gitleaks.

---

### Task 1: Export the knowledge-graph boundary

**Files:**
- Keep: `src/material_graph/knowledge/**` except agent/task modules
- Keep: `config/knowledge/**` except agent-subgraph profiles
- Keep: `migrations/knowledge_*.sql`, `migrations/age_0001*.sql`
- Keep: knowledge-build/import/layout scripts and focused tests
- Remove: workflow, API/SSE, memory, UI, domain packs, prediction, agent views, task subgraphs, and task-evidence execution

- [ ] Export tracked files from commit `6236ed195983e6e34543eb98b39f1a82990f0e7d` with `git archive`.
- [ ] Delete `agent_views.py`, `gap_analysis.py`, `literature_search.py`, `task_subgraph.py`, `textbook_task_evidence.py`, and `worker_factory.py` from the exported knowledge package.
- [ ] Replace both package `__init__.py` files with knowledge-graph-only module boundaries.

### Task 2: Rewrite repository documentation and packaging

**Files:**
- Create: `README.md`
- Create: `docs/knowledge-graph-pipeline.md`
- Create: `docs/public-release-boundary.md`
- Create: `PUBLICATION_MANIFEST.md`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

- [ ] Document only source discovery, parsing, extraction, review, AGE/LightRAG persistence, admission, audit, and layout.
- [ ] Exclude every command and description that runs an agent workflow, API, UI, memory service, prediction, or experiment planner.
- [ ] Configure the package entry point as `material-graph-knowledge = material_graph.knowledge.cli:main`.

### Task 3: Verify on the Nantong server

- [ ] Run `python -m compileall -q src scripts` with bytecode redirected to container `/tmp`.
- [ ] Run `ruff check --no-cache src tests scripts`.
- [ ] Run the full exported pytest suite in a disposable container with `--network none`.
- [ ] Run Gitleaks against the complete new tree and confirm no leak.
- [ ] Confirm `git grep` finds no `workflow`, `agent_views`, `TaskSubgraph`, FastAPI, SSE, Streamlit, or memory implementation files.

### Task 4: Replace public history safely

- [ ] Create a new root commit in `/home/cyy/dhu-zh-workspace/omnimat/worktrees/material-knowledge-graph-public`.
- [ ] Verify the current remote main SHA is exactly `6236ed195983e6e34543eb98b39f1a82990f0e7d`.
- [ ] Push with `--force-with-lease=refs/heads/main:6236ed195983e6e34543eb98b39f1a82990f0e7d` using the repository-local `fafasco16` SSH key.
- [ ] Verify local HEAD equals remote main, the GitHub workflow succeeds, and `CYJTHUDA` retains repository administrator permission.
