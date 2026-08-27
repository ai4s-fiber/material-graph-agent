# Material Knowledge-Graph Construction Pipeline

## 1. Source catalog and selection

`catalog.py`, `policy.py`, and `selection.py` create versioned source identities and apply bounded corpus policies. `remote_reader.py` and `connectors/synology*` enforce read-only range and streaming contracts. Credentials are runtime-only and are never serialized into catalog records.

## 2. Parsing and evidence fragments

`mineru_client.py`, `ingestion.py`, `processing.py`, `retention.py`, and `spool.py` turn selected objects into provenance-bound `EvidenceFragment` records. Deterministic idempotency keys, checkpoints, cancellation, size budgets, and cleanup prevent partial or duplicated ingestion.

## 3. Typed fact extraction

`extraction.py`, `openai_fact_extractor.py`, and `facts.py` validate structured provider output before it becomes a `FactBatch`. Duplicate JSON keys, non-finite values, unsafe locations, missing evidence, invalid units, and over-budget payloads fail closed.

## 4. Review and approved graph writes

`reviewed_graph.py`, `jobs.py`, `postgres_jobs.py`, and `age_writer.py` enforce the human-review boundary. Pending or rejected batches cannot create graph mutations. Approved outbox jobs write parameterized, idempotent node and edge projections to PostgreSQL/Apache AGE.

## 5. LightRAG and portable generations

`lightrag_*` modules validate non-secret bindings, storage namespaces, source mappings, and least-privilege behavior. The `textbook_*` modules build raw extraction records, canonical custom-KG streams, portable embedding archives, strict provenance contracts, and server-admission bundles.

Build order:

```text
corpus preparation
  -> raw graph extraction
  -> entity/relation merge and custom-KG bundle
  -> generation-bound embedding archive
  -> provenance deployment bundle
  -> admission validation
  -> provenance import
  -> LightRAG graph/KV/vector import
```

A provider, model, dimension, workspace, generation, count, or digest mismatch blocks admission.

## 6. Audit and layout

`audit_material_graph.py` validates canonical entities and relations. `build_material_graph_layout.py` creates a separate undirected weighted layout graph for community detection and coordinates; it never replaces the directed semantic graph. The verifier and publisher emit browser assets without source fragments or relation descriptions.

## 7. Verification commands

```bash
pytest -q tests/test_ingestion_pipeline.py tests/test_fact_extraction_pipeline.py
pytest -q tests/test_reviewed_graph.py tests/test_age_graph_writer.py
pytest -q tests/test_textbook_graph_bundle.py tests/test_textbook_embedding_bundle.py
pytest -q tests/test_textbook_server_admission.py tests/test_material_graph_layout_builder.py
```

Real AGE integration tests must use a disposable database and an explicit opt-in environment flag.
