"""Stable provider-neutral contracts for material knowledge-graph engineering."""

from .age_writer import GraphWriteApproval, PostgresAGEGlobalKnowledgeGraphWriter
from .catalog import SourceCatalogRepository, build_source_version_key
from .extraction import EvidenceFactExtractionPipeline
from .facts import (
    EntityRef,
    FactBatch,
    GlobalKnowledgeGraphWriter,
    PropertyObservation,
)
from .ingestion import (
    CheckpointRepository,
    EvidenceRepository,
    KnowledgeIngestionPipeline,
    build_ingestion_idempotency_key,
)
from .manifest import MetadataCursor, MetadataManifestIngestor
from .models import EvidenceFragment
from .policy import CorpusPolicy
from .postgres import (
    PostgresCheckpointRepository,
    PostgresEvidenceRepository,
    PostgresSourceCatalogRepository,
)
from .processing import ProcessingCheckpoint
from .remote_reader import RemoteRangeContractError, RemoteSourceReader
from .selection import SelectionPolicy
from .service import KnowledgeCanaryService

__all__ = [
    "CheckpointRepository",
    "CorpusPolicy",
    "EvidenceFactExtractionPipeline",
    "EntityRef",
    "EvidenceFragment",
    "EvidenceRepository",
    "FactBatch",
    "GlobalKnowledgeGraphWriter",
    "GraphWriteApproval",
    "KnowledgeCanaryService",
    "KnowledgeIngestionPipeline",
    "MetadataCursor",
    "MetadataManifestIngestor",
    "PostgresAGEGlobalKnowledgeGraphWriter",
    "PostgresCheckpointRepository",
    "PostgresEvidenceRepository",
    "PostgresSourceCatalogRepository",
    "ProcessingCheckpoint",
    "PropertyObservation",
    "RemoteRangeContractError",
    "RemoteSourceReader",
    "SelectionPolicy",
    "SourceCatalogRepository",
    "build_ingestion_idempotency_key",
    "build_source_version_key",
]
