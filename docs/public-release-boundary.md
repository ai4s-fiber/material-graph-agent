# Public Release Boundary

## Included

- `src/material_graph/knowledge` ingestion, extraction, review, AGE/LightRAG, textbook build/import, audit, and layout code;
- non-secret knowledge policies and provider bindings;
- knowledge and AGE migrations;
- focused scripts, tests, third-party notices, and deterministic synthetic fixtures created inside tests.

## Excluded

- task agents, orchestration graphs, role-specific views, task subgraphs, API/SSE, UI, memory, prediction, experiments, and managed compute;
- production operations, host inventories, cutover evidence, account systems, and frontend source;
- credentials, `.env`, raw documents, MinerU archives, fragment streams, vector binaries, GraphML output, databases, and runtime artifacts.

## Data statement

Tests use synthetic engineering fixtures. They are not experimental measurements, scientific conclusions, or process recommendations.

## License

Public visibility does not grant permission to use, copy, modify, or distribute OmniMat-owned code. See the root `LICENSE` and file-specific third-party licenses.
