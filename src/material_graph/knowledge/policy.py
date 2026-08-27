"""Non-sensitive logical corpus and storage policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


FORBIDDEN_CONFIGURATION_SEGMENTS = {
    "auth",
    "cookie",
    "credential",
    "did",
    "endpoint",
    "host",
    "address",
    "quickconnect",
    "secret",
    "session",
    "sid",
    "username",
    "password",
    "api_key",
    "token",
}
FORBIDDEN_CONFIGURATION_MARKERS = {
    "accesstoken",
    "apikey",
    "clientsecret",
    "credential",
    "deviceid",
    "devicetoken",
    "password",
    "passwd",
    "privatekey",
    "quickconnect",
    "refreshtoken",
    "sessioncookie",
    "sessionsid",
    "synotoken",
}


def _reject_forbidden_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            compact = normalized.replace("_", "")
            segments = set(normalized.split("_"))
            if segments.intersection(FORBIDDEN_CONFIGURATION_SEGMENTS) or any(
                marker in compact for marker in FORBIDDEN_CONFIGURATION_MARKERS
            ):
                raise ValueError(f"forbidden configuration field: {normalized}")
            _reject_forbidden_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_fields(child)


class SpoolPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_total_bytes: int = Field(gt=0)
    max_object_bytes: int = Field(gt=0)
    max_active_objects: int = Field(gt=0)
    abandoned_ttl_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def object_must_fit_spool(self) -> "SpoolPolicy":
        if self.max_object_bytes > self.max_total_bytes:
            raise ValueError("spool object limit exceeds total spool capacity")
        return self


class CorpusSourcePolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    root_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(min_length=1)
    verified_size_gb: float = Field(ge=0)
    pdf_count: int | None = Field(default=None, ge=0)
    file_count: int | None = Field(default=None, ge=0)
    literature_size_gb: float | None = Field(default=None, ge=0)
    patent_size_gb: float | None = Field(default=None, ge=0)
    usable_size_gb: float | None = Field(default=None, ge=0)
    excluded_process_size_gb: float | None = Field(default=None, ge=0)
    physical_material_type_count: int | None = Field(default=None, ge=0)
    physical_material_type_ids: list[str] = Field(default_factory=list)
    deduplicated_record_count: int | None = Field(default=None, ge=0)
    existing_pdf_record_count: int | None = Field(default=None, ge=0)
    no_source_record_count: int | None = Field(default=None, ge=0)
    metadata_formats: list[str] = Field(default_factory=list)
    deduplicate_against: str | None = None

    @model_validator(mode="after")
    def material_type_count_matches_ids(self) -> "CorpusSourcePolicy":
        if (
            self.physical_material_type_count is not None
            and self.physical_material_type_ids
            and self.physical_material_type_count != len(self.physical_material_type_ids)
        ):
            raise ValueError("physical material type count does not match IDs")
        return self


class CorpusPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    server_capacity_gb: float = Field(gt=0)
    minimum_free_bytes: int = Field(gt=0)
    hard_stop_free_bytes: int = Field(gt=0)
    derived_data_target_bytes: int = Field(gt=0)
    derived_data_hard_cap_bytes: int = Field(gt=0)
    filesystem_alert_ratio: float = Field(gt=0, lt=1)
    filesystem_stop_ratio: float = Field(gt=0, lt=1)
    spool: SpoolPolicy
    sources: list[CorpusSourcePolicy] = Field(min_length=1)

    @classmethod
    def load(cls, path: str | Path) -> "CorpusPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        _reject_forbidden_fields(payload)
        return cls.model_validate(payload)

    def source(self, root_id: str) -> CorpusSourcePolicy:
        for source in self.sources:
            if source.root_id == root_id:
                return source
        raise KeyError(f"unknown corpus root: {root_id}")

    @model_validator(mode="after")
    def validate_capacity_and_roots(self) -> "CorpusPolicy":
        if self.hard_stop_free_bytes > self.minimum_free_bytes:
            raise ValueError("hard stop free space must not exceed normal minimum free space")
        if self.derived_data_target_bytes > self.derived_data_hard_cap_bytes:
            raise ValueError("derived-data target exceeds its hard cap")
        if self.filesystem_alert_ratio > self.filesystem_stop_ratio:
            raise ValueError("filesystem alert ratio must not exceed stop ratio")

        server_bytes = int(self.server_capacity_gb * 1024**3)
        if self.derived_data_hard_cap_bytes + self.minimum_free_bytes > server_bytes:
            raise ValueError("derived-data hard cap leaves insufficient free space")
        if self.spool.max_total_bytes > server_bytes - self.minimum_free_bytes:
            raise ValueError("spool quota exceeds available server capacity")

        root_ids = [source.root_id for source in self.sources]
        if len(root_ids) != len(set(root_ids)):
            raise ValueError("corpus root IDs must be unique")
        return self
