"""Durable cross-process admission and circuit coordination for shared providers."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from hashlib import sha256
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_STATE_COLUMNS = """
provider_scope, policy_fingerprint, bulk_limit, bulk_successes,
quota_window_started_at, requests_used, tokens_reserved, last_429_window,
circuit_state, consecutive_failures, circuit_open_until, next_lease_token
""".strip()
_ENSURE_STATE = """
INSERT INTO provider_runtime_state(provider_scope, policy_fingerprint, bulk_limit)
VALUES (%s, %s, %s)
ON CONFLICT (provider_scope) DO NOTHING
"""
_LOCK_STATE = f"""
SELECT {_STATE_COLUMNS}, now() AS database_now
FROM provider_runtime_state
WHERE provider_scope = %s
FOR UPDATE
"""
_WRITE_STATE = """
UPDATE provider_runtime_state
SET bulk_limit = %s, bulk_successes = %s, quota_window_started_at = %s,
    requests_used = %s, tokens_reserved = %s, last_429_window = %s,
    circuit_state = %s, consecutive_failures = %s, circuit_open_until = %s,
    next_lease_token = %s, updated_at = now()
WHERE provider_scope = %s AND policy_fingerprint = %s
RETURNING provider_scope
"""
_DELETE_EXPIRED = """
DELETE FROM provider_runtime_claims
WHERE provider_scope = %s AND lease_until <= now()
RETURNING half_open_probe
"""
_COUNT_ACTIVE = """
SELECT workload, count(*) AS active_count
FROM provider_runtime_claims
WHERE provider_scope = %s AND lease_until > now()
GROUP BY workload
"""
_HALF_OPEN_ACTIVE = """
SELECT half_open_probe FROM provider_runtime_claims
WHERE provider_scope = %s AND half_open_probe AND lease_until > now()
"""
_INSERT_CLAIM = """
INSERT INTO provider_runtime_claims(
    claim_id, provider_scope, policy_fingerprint, owner_id, workload,
    lease_token, estimated_tokens, half_open_probe, lease_until
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now() + (%s * INTERVAL '1 second'))
RETURNING claim_id, provider_scope, policy_fingerprint, owner_id, workload,
          lease_token, estimated_tokens, half_open_probe, claimed_at, lease_until
"""
_VERIFY_CLAIM = """
SELECT claim_id FROM provider_runtime_claims
WHERE claim_id = %s AND provider_scope = %s AND owner_id = %s
  AND lease_token = %s AND policy_fingerprint = %s AND lease_until > now()
FOR UPDATE
"""
_RELEASE_CLAIM = """
DELETE FROM provider_runtime_claims
WHERE claim_id = %s AND provider_scope = %s AND owner_id = %s
  AND lease_token = %s AND policy_fingerprint = %s
RETURNING claim_id
"""


class ProviderWorkload(str, Enum):
    INTERACTIVE = "interactive"
    BULK = "bulk"


class ProviderRuntimePolicy(BaseModel):
    """Immutable provider budget shared by every API and worker process."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    total_slots: int = Field(default=16, ge=3, le=512)
    reserved_interactive_slots: int = Field(default=2, ge=1, le=128)
    bulk_initial_slots: int = Field(default=4, ge=1, le=512)
    bulk_hard_max: int = Field(default=12, ge=1, le=512)
    requests_per_minute: int = Field(default=120, ge=2, le=1_000_000)
    tokens_per_minute: int = Field(default=500_000, ge=2, le=2_000_000_000)
    reserved_interactive_requests: int = Field(default=8, ge=1)
    reserved_interactive_tokens: int = Field(default=32_000, ge=1)
    lease_ttl_seconds: int = Field(default=960, ge=5, le=7200)
    acquire_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    poll_interval_seconds: float = Field(default=0.05, ge=0.005, le=1)
    recovery_successes: int = Field(default=20, ge=1, le=10_000)
    circuit_failure_threshold: int = Field(default=5, ge=1, le=100)
    circuit_open_seconds: int = Field(default=30, ge=1, le=3600)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    @model_validator(mode="after")
    def validate_reservations(self) -> "ProviderRuntimePolicy":
        bulk_capacity = self.total_slots - self.reserved_interactive_slots
        if not self.bulk_initial_slots <= self.bulk_hard_max <= bulk_capacity:
            raise ValueError("bulk capacity consumes interactive reservation")
        if self.reserved_interactive_requests >= self.requests_per_minute:
            raise ValueError("invalid request reservation")
        if self.reserved_interactive_tokens >= self.tokens_per_minute:
            raise ValueError("invalid token reservation")
        return self

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "ProviderRuntimePolicy":
        values = os.environ if environment is None else environment

        def value(name: str, default: int | float, legacy: str | None = None) -> str:
            raw = values.get(name, "").strip()
            if not raw and legacy:
                raw = values.get(legacy, "").strip()
            return raw or str(default)

        return cls(
            total_slots=int(
                value("MATERIAL_GRAPH_PROVIDER_TOTAL_SLOTS", 16, "INGEST_LLM_SHARED_HARD_MAX")
            ),
            reserved_interactive_slots=int(
                value(
                    "MATERIAL_GRAPH_PROVIDER_RESERVED_INTERACTIVE_SLOTS",
                    2,
                    "INGEST_LLM_INTERACTIVE_RESERVED",
                )
            ),
            bulk_initial_slots=int(
                value("MATERIAL_GRAPH_PROVIDER_BULK_INITIAL_SLOTS", 4, "INGEST_LLM_BULK_INITIAL")
            ),
            bulk_hard_max=int(
                value("MATERIAL_GRAPH_PROVIDER_BULK_HARD_MAX", 12, "INGEST_LLM_BULK_HARD_MAX")
            ),
            requests_per_minute=int(value("MATERIAL_GRAPH_PROVIDER_REQUESTS_PER_MINUTE", 120)),
            tokens_per_minute=int(value("MATERIAL_GRAPH_PROVIDER_TOKENS_PER_MINUTE", 500_000)),
            reserved_interactive_requests=int(
                value("MATERIAL_GRAPH_PROVIDER_RESERVED_INTERACTIVE_REQUESTS", 8)
            ),
            reserved_interactive_tokens=int(
                value("MATERIAL_GRAPH_PROVIDER_RESERVED_INTERACTIVE_TOKENS", 32_000)
            ),
            lease_ttl_seconds=int(value("MATERIAL_GRAPH_PROVIDER_LEASE_TTL_SECONDS", 960)),
            acquire_timeout_seconds=float(
                value("MATERIAL_GRAPH_PROVIDER_ACQUIRE_TIMEOUT_SECONDS", 30)
            ),
            poll_interval_seconds=float(
                value("MATERIAL_GRAPH_PROVIDER_POLL_INTERVAL_SECONDS", 0.05)
            ),
            recovery_successes=int(value("MATERIAL_GRAPH_PROVIDER_RECOVERY_SUCCESSES", 20)),
            circuit_failure_threshold=int(
                value("MATERIAL_GRAPH_PROVIDER_CIRCUIT_FAILURE_THRESHOLD", 5)
            ),
            circuit_open_seconds=int(value("MATERIAL_GRAPH_PROVIDER_CIRCUIT_OPEN_SECONDS", 30)),
            rate_limit_window_seconds=int(
                value("MATERIAL_GRAPH_PROVIDER_RATE_LIMIT_WINDOW_SECONDS", 60)
            ),
        )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def provider_scope_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    provider_identity: str | None = None,
) -> str:
    values = os.environ if environment is None else environment
    configured = values.get("MATERIAL_GRAPH_PROVIDER_SCOPE", "").strip()
    if configured:
        scope = configured
    elif provider_identity:
        digest = sha256(provider_identity.strip().encode("utf-8")).hexdigest()[:24]
        scope = f"openai-compatible:{digest}"
    else:
        scope = "openai-compatible:gpt56-primary"
    if _SAFE_ID.fullmatch(scope) is None:
        raise ValueError("invalid provider scope")
    return scope


class ProviderCoordinationError(RuntimeError):
    def __init__(
        self, code: str, *, retryable: bool, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class ProviderLeaseLost(ProviderCoordinationError):
    def __init__(self) -> None:
        super().__init__("provider_coordination.lease_lost", retryable=True)


class ProviderRuntimePersistenceError(ProviderCoordinationError):
    def __init__(self) -> None:
        super().__init__("provider_coordination.persistence_unavailable", retryable=True)


@dataclass(frozen=True, slots=True)
class ProviderLease:
    claim_id: UUID
    provider_scope: str
    policy_fingerprint: str
    owner_id: str
    workload: ProviderWorkload
    lease_token: int
    estimated_tokens: int
    half_open_probe: bool
    claimed_at: datetime
    lease_until: datetime


@dataclass(frozen=True, slots=True)
class ProviderAdmission:
    outcome: str
    lease: ProviderLease | None = None
    retry_after_seconds: float | None = None


@dataclass(slots=True)
class _RuntimeState:
    provider_scope: str
    policy_fingerprint: str
    bulk_limit: int
    bulk_successes: int
    quota_window_started_at: datetime
    requests_used: int
    tokens_reserved: int
    last_429_window: int | None
    circuit_state: str
    consecutive_failures: int
    circuit_open_until: datetime | None
    next_lease_token: int


class ProviderMetrics(Protocol):
    def inc(self, name: str, value: float = 1, **labels: str) -> None: ...
    def observe(self, name: str, value: float, **labels: str) -> None: ...


class _SyncCursor(Protocol):
    def fetchone(self) -> Mapping[str, Any] | None: ...
    def fetchall(self) -> Sequence[Mapping[str, Any]]: ...


class _SyncConnection(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> _SyncCursor: ...
    def transaction(self) -> AbstractContextManager[Any]: ...


class SyncConnectionPool(Protocol):
    def connection(self) -> AbstractContextManager[_SyncConnection]: ...


class _AsyncCursor(Protocol):
    async def fetchone(self) -> Mapping[str, Any] | None: ...
    async def fetchall(self) -> Sequence[Mapping[str, Any]]: ...


class _AsyncConnection(Protocol):
    async def execute(self, query: str, params: Sequence[Any] | None = None) -> _AsyncCursor: ...
    def transaction(self) -> AbstractAsyncContextManager[Any]: ...


class AsyncConnectionPool(Protocol):
    def connection(self) -> AbstractAsyncContextManager[_AsyncConnection]: ...


class SyncProviderStore(Protocol):
    def try_acquire(
        self,
        *,
        provider_scope: str,
        policy: ProviderRuntimePolicy,
        owner_id: str,
        workload: ProviderWorkload,
        estimated_tokens: int,
    ) -> ProviderAdmission: ...
    def release(self, lease: ProviderLease) -> None: ...
    def record_success(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None: ...
    def record_failure(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None: ...
    def record_429(
        self, *, provider_scope: str, workload: ProviderWorkload, policy: ProviderRuntimePolicy
    ) -> None: ...
    def snapshot(self, *, provider_scope: str, policy: ProviderRuntimePolicy) -> dict[str, Any]: ...


class AsyncProviderStore(Protocol):
    async def try_acquire(
        self,
        *,
        provider_scope: str,
        policy: ProviderRuntimePolicy,
        owner_id: str,
        workload: ProviderWorkload,
        estimated_tokens: int,
    ) -> ProviderAdmission: ...
    async def release(self, lease: ProviderLease) -> None: ...
    async def record_success(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None: ...
    async def record_failure(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None: ...
    async def record_429(
        self, *, provider_scope: str, workload: ProviderWorkload, policy: ProviderRuntimePolicy
    ) -> None: ...
    async def snapshot(
        self, *, provider_scope: str, policy: ProviderRuntimePolicy
    ) -> dict[str, Any]: ...


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderRuntimePersistenceError()
    return value


def _state_from_row(row: Mapping[str, Any]) -> _RuntimeState:
    return _RuntimeState(
        provider_scope=str(row["provider_scope"]),
        policy_fingerprint=str(row["policy_fingerprint"]),
        bulk_limit=int(row["bulk_limit"]),
        bulk_successes=int(row["bulk_successes"]),
        quota_window_started_at=row["quota_window_started_at"],
        requests_used=int(row["requests_used"]),
        tokens_reserved=int(row["tokens_reserved"]),
        last_429_window=None if row.get("last_429_window") is None else int(row["last_429_window"]),
        circuit_state=str(row["circuit_state"]),
        consecutive_failures=int(row["consecutive_failures"]),
        circuit_open_until=row.get("circuit_open_until"),
        next_lease_token=int(row["next_lease_token"]),
    )


def _lease_from_row(row: Mapping[str, Any]) -> ProviderLease:
    return ProviderLease(
        claim_id=UUID(str(row["claim_id"])),
        provider_scope=str(row["provider_scope"]),
        policy_fingerprint=str(row["policy_fingerprint"]),
        owner_id=str(row["owner_id"]),
        workload=ProviderWorkload(str(row["workload"])),
        lease_token=int(row["lease_token"]),
        estimated_tokens=int(row["estimated_tokens"]),
        half_open_probe=bool(row["half_open_probe"]),
        claimed_at=row["claimed_at"],
        lease_until=row["lease_until"],
    )


def _state_params(state: _RuntimeState) -> tuple[object, ...]:
    return (
        state.bulk_limit,
        state.bulk_successes,
        state.quota_window_started_at,
        state.requests_used,
        state.tokens_reserved,
        state.last_429_window,
        state.circuit_state,
        state.consecutive_failures,
        state.circuit_open_until,
        state.next_lease_token,
        state.provider_scope,
        state.policy_fingerprint,
    )


def _validate_identity(provider_scope: str, owner_id: str) -> None:
    if _SAFE_ID.fullmatch(provider_scope) is None or _SAFE_ID.fullmatch(owner_id) is None:
        raise ValueError("invalid provider coordination identity")


def _validate_attempt(workload: ProviderWorkload | str, estimated_tokens: int) -> ProviderWorkload:
    selected = ProviderWorkload(workload)
    if isinstance(estimated_tokens, bool) or estimated_tokens < 1:
        raise ValueError("estimated_tokens must be positive")
    return selected


def _check_policy(state: _RuntimeState, policy: ProviderRuntimePolicy) -> None:
    if state.policy_fingerprint != policy.fingerprint:
        raise ProviderCoordinationError("provider_coordination.policy_mismatch", retryable=False)


def _advance_state(
    state: _RuntimeState,
    *,
    now: datetime,
    policy: ProviderRuntimePolicy,
    expired_half_open: bool,
) -> None:
    if now - state.quota_window_started_at >= timedelta(minutes=1):
        state.quota_window_started_at = now
        state.requests_used = 0
        state.tokens_reserved = 0
    if expired_half_open and state.circuit_state == "half_open":
        state.circuit_state = "open"
        state.circuit_open_until = now + timedelta(seconds=policy.circuit_open_seconds)
        state.consecutive_failures = policy.circuit_failure_threshold
    if (
        state.circuit_state == "open"
        and state.circuit_open_until is not None
        and state.circuit_open_until <= now
    ):
        state.circuit_state = "half_open"
        state.circuit_open_until = None


def _admission(
    state: _RuntimeState,
    *,
    now: datetime,
    counts: Mapping[ProviderWorkload, int],
    policy: ProviderRuntimePolicy,
    workload: ProviderWorkload,
    estimated_tokens: int,
    half_open_active: bool,
) -> ProviderAdmission:
    if state.circuit_state == "open":
        retry = 0.0
        if state.circuit_open_until is not None:
            retry = max(0.0, (state.circuit_open_until - now).total_seconds())
        return ProviderAdmission("circuit_open", retry_after_seconds=retry)
    if state.circuit_state == "half_open" and half_open_active:
        return ProviderAdmission("wait")
    interactive = counts.get(ProviderWorkload.INTERACTIVE, 0)
    bulk = counts.get(ProviderWorkload.BULK, 0)
    if interactive + bulk >= policy.total_slots:
        return ProviderAdmission("wait")
    if workload is ProviderWorkload.BULK:
        bulk_cap = min(
            state.bulk_limit,
            policy.bulk_hard_max,
            policy.total_slots - policy.reserved_interactive_slots,
        )
        if bulk >= bulk_cap:
            return ProviderAdmission("wait")
        request_cap = policy.requests_per_minute - policy.reserved_interactive_requests
        token_cap = policy.tokens_per_minute - policy.reserved_interactive_tokens
    else:
        request_cap = policy.requests_per_minute
        token_cap = policy.tokens_per_minute
    if state.requests_used + 1 > request_cap:
        return ProviderAdmission("wait")
    if state.tokens_reserved + estimated_tokens > token_cap:
        return ProviderAdmission("wait")
    return ProviderAdmission("ready")


def _record_success_state(
    state: _RuntimeState, lease: ProviderLease, policy: ProviderRuntimePolicy
) -> None:
    if lease.half_open_probe:
        state.circuit_state = "closed"
        state.circuit_open_until = None
        state.consecutive_failures = 0
    elif state.circuit_state == "closed":
        state.consecutive_failures = 0
    if lease.workload is ProviderWorkload.BULK:
        state.bulk_successes += 1
        if state.bulk_successes >= policy.recovery_successes:
            state.bulk_limit = min(policy.bulk_hard_max, state.bulk_limit + 1)
            state.bulk_successes = 0


def _record_failure_state(
    state: _RuntimeState,
    lease: ProviderLease,
    *,
    now: datetime,
    policy: ProviderRuntimePolicy,
) -> None:
    failures = state.consecutive_failures + 1
    if lease.half_open_probe or failures >= policy.circuit_failure_threshold:
        state.circuit_state = "open"
        state.circuit_open_until = now + timedelta(seconds=policy.circuit_open_seconds)
        state.consecutive_failures = failures
    elif state.circuit_state == "closed":
        state.consecutive_failures = failures


def _record_429_state(
    state: _RuntimeState, *, now: datetime, policy: ProviderRuntimePolicy
) -> bool:
    window = math.floor(now.timestamp() / policy.rate_limit_window_seconds)
    if state.last_429_window == window:
        return False
    state.last_429_window = window
    state.bulk_limit = max(1, state.bulk_limit // 2)
    state.bulk_successes = 0
    return True


def _snapshot(state: _RuntimeState, counts: Mapping[ProviderWorkload, int]) -> dict[str, Any]:
    return {
        "provider_scope": state.provider_scope,
        "policy_fingerprint": state.policy_fingerprint,
        "active_interactive": counts.get(ProviderWorkload.INTERACTIVE, 0),
        "active_bulk": counts.get(ProviderWorkload.BULK, 0),
        "bulk_limit": state.bulk_limit,
        "bulk_successes": state.bulk_successes,
        "requests_used": state.requests_used,
        "tokens_reserved": state.tokens_reserved,
        "circuit_state": state.circuit_state,
        "consecutive_failures": state.consecutive_failures,
        "circuit_open_until": state.circuit_open_until,
    }


class PostgresSyncProviderStore:
    """Synchronous PostgreSQL provider state repository."""

    def __init__(self, pool: SyncConnectionPool) -> None:
        self._pool = pool

    @staticmethod
    def _write(connection: _SyncConnection, state: _RuntimeState) -> None:
        if connection.execute(_WRITE_STATE, _state_params(state)).fetchone() is None:
            raise ProviderRuntimePersistenceError()

    def try_acquire(
        self,
        *,
        provider_scope: str,
        policy: ProviderRuntimePolicy,
        owner_id: str,
        workload: ProviderWorkload,
        estimated_tokens: int,
    ) -> ProviderAdmission:
        try:
            with self._pool.connection() as connection:
                with connection.transaction():
                    connection.execute(
                        _ENSURE_STATE,
                        (provider_scope, policy.fingerprint, policy.bulk_initial_slots),
                    )
                    raw = connection.execute(_LOCK_STATE, (provider_scope,)).fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    expired = connection.execute(_DELETE_EXPIRED, (provider_scope,)).fetchall()
                    now = row["database_now"]
                    _advance_state(
                        state,
                        now=now,
                        policy=policy,
                        expired_half_open=any(bool(item["half_open_probe"]) for item in expired),
                    )
                    active_rows = connection.execute(_COUNT_ACTIVE, (provider_scope,)).fetchall()
                    counts = {
                        ProviderWorkload(str(item["workload"])): int(item["active_count"])
                        for item in active_rows
                    }
                    half_open_active = any(
                        bool(item["half_open_probe"])
                        for item in connection.execute(
                            _HALF_OPEN_ACTIVE, (provider_scope,)
                        ).fetchall()
                    )
                    decision = _admission(
                        state,
                        now=now,
                        counts=counts,
                        policy=policy,
                        workload=workload,
                        estimated_tokens=estimated_tokens,
                        half_open_active=half_open_active,
                    )
                    if decision.outcome != "ready":
                        self._write(connection, state)
                        return decision
                    state.next_lease_token += 1
                    state.requests_used += 1
                    state.tokens_reserved += estimated_tokens
                    self._write(connection, state)
                    raw = connection.execute(
                        _INSERT_CLAIM,
                        (
                            uuid4(),
                            provider_scope,
                            policy.fingerprint,
                            owner_id,
                            workload.value,
                            state.next_lease_token,
                            estimated_tokens,
                            state.circuit_state == "half_open",
                            policy.lease_ttl_seconds,
                        ),
                    ).fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    return ProviderAdmission("acquired", lease=_lease_from_row(_mapping(raw)))
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    def release(self, lease: ProviderLease) -> None:
        try:
            with self._pool.connection() as connection:
                connection.execute(
                    _RELEASE_CLAIM,
                    (
                        lease.claim_id,
                        lease.provider_scope,
                        lease.owner_id,
                        lease.lease_token,
                        lease.policy_fingerprint,
                    ),
                ).fetchone()
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    def _outcome(
        self,
        lease: ProviderLease,
        policy: ProviderRuntimePolicy,
        update: Callable[[_RuntimeState, datetime], None],
    ) -> None:
        try:
            with self._pool.connection() as connection:
                with connection.transaction():
                    raw = connection.execute(_LOCK_STATE, (lease.provider_scope,)).fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    if (
                        connection.execute(
                            _VERIFY_CLAIM,
                            (
                                lease.claim_id,
                                lease.provider_scope,
                                lease.owner_id,
                                lease.lease_token,
                                lease.policy_fingerprint,
                            ),
                        ).fetchone()
                        is None
                    ):
                        raise ProviderLeaseLost()
                    update(state, row["database_now"])
                    self._write(connection, state)
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    def record_success(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        self._outcome(lease, policy, lambda state, _: _record_success_state(state, lease, policy))

    def record_failure(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        self._outcome(
            lease,
            policy,
            lambda state, now: _record_failure_state(state, lease, now=now, policy=policy),
        )

    def record_429(
        self, *, provider_scope: str, workload: ProviderWorkload, policy: ProviderRuntimePolicy
    ) -> None:
        del workload
        try:
            with self._pool.connection() as connection:
                with connection.transaction():
                    raw = connection.execute(_LOCK_STATE, (provider_scope,)).fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    _record_429_state(state, now=row["database_now"], policy=policy)
                    self._write(connection, state)
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    def snapshot(self, *, provider_scope: str, policy: ProviderRuntimePolicy) -> dict[str, Any]:
        try:
            with self._pool.connection() as connection:
                with connection.transaction():
                    connection.execute(
                        _ENSURE_STATE,
                        (provider_scope, policy.fingerprint, policy.bulk_initial_slots),
                    )
                    raw = connection.execute(_LOCK_STATE, (provider_scope,)).fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    expired = connection.execute(_DELETE_EXPIRED, (provider_scope,)).fetchall()
                    _advance_state(
                        state,
                        now=row["database_now"],
                        policy=policy,
                        expired_half_open=any(bool(item["half_open_probe"]) for item in expired),
                    )
                    active_rows = connection.execute(_COUNT_ACTIVE, (provider_scope,)).fetchall()
                    self._write(connection, state)
                    counts = {
                        ProviderWorkload(str(item["workload"])): int(item["active_count"])
                        for item in active_rows
                    }
                    return _snapshot(state, counts)
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None


class PostgresAsyncProviderStore:
    """Asynchronous PostgreSQL provider state repository for workers."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    @staticmethod
    async def _write(connection: _AsyncConnection, state: _RuntimeState) -> None:
        cursor = await connection.execute(_WRITE_STATE, _state_params(state))
        if await cursor.fetchone() is None:
            raise ProviderRuntimePersistenceError()

    async def try_acquire(
        self,
        *,
        provider_scope: str,
        policy: ProviderRuntimePolicy,
        owner_id: str,
        workload: ProviderWorkload,
        estimated_tokens: int,
    ) -> ProviderAdmission:
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    await connection.execute(
                        _ENSURE_STATE,
                        (provider_scope, policy.fingerprint, policy.bulk_initial_slots),
                    )
                    cursor = await connection.execute(_LOCK_STATE, (provider_scope,))
                    raw = await cursor.fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    cursor = await connection.execute(_DELETE_EXPIRED, (provider_scope,))
                    expired = await cursor.fetchall()
                    now = row["database_now"]
                    _advance_state(
                        state,
                        now=now,
                        policy=policy,
                        expired_half_open=any(bool(item["half_open_probe"]) for item in expired),
                    )
                    cursor = await connection.execute(_COUNT_ACTIVE, (provider_scope,))
                    active_rows = await cursor.fetchall()
                    counts = {
                        ProviderWorkload(str(item["workload"])): int(item["active_count"])
                        for item in active_rows
                    }
                    cursor = await connection.execute(_HALF_OPEN_ACTIVE, (provider_scope,))
                    half_open_active = any(
                        bool(item["half_open_probe"]) for item in await cursor.fetchall()
                    )
                    decision = _admission(
                        state,
                        now=now,
                        counts=counts,
                        policy=policy,
                        workload=workload,
                        estimated_tokens=estimated_tokens,
                        half_open_active=half_open_active,
                    )
                    if decision.outcome != "ready":
                        await self._write(connection, state)
                        return decision
                    state.next_lease_token += 1
                    state.requests_used += 1
                    state.tokens_reserved += estimated_tokens
                    await self._write(connection, state)
                    cursor = await connection.execute(
                        _INSERT_CLAIM,
                        (
                            uuid4(),
                            provider_scope,
                            policy.fingerprint,
                            owner_id,
                            workload.value,
                            state.next_lease_token,
                            estimated_tokens,
                            state.circuit_state == "half_open",
                            policy.lease_ttl_seconds,
                        ),
                    )
                    raw = await cursor.fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    return ProviderAdmission("acquired", lease=_lease_from_row(_mapping(raw)))
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    async def release(self, lease: ProviderLease) -> None:
        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    _RELEASE_CLAIM,
                    (
                        lease.claim_id,
                        lease.provider_scope,
                        lease.owner_id,
                        lease.lease_token,
                        lease.policy_fingerprint,
                    ),
                )
                await cursor.fetchone()
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    async def _outcome(
        self,
        lease: ProviderLease,
        policy: ProviderRuntimePolicy,
        update: Callable[[_RuntimeState, datetime], None],
    ) -> None:
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(_LOCK_STATE, (lease.provider_scope,))
                    raw = await cursor.fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    cursor = await connection.execute(
                        _VERIFY_CLAIM,
                        (
                            lease.claim_id,
                            lease.provider_scope,
                            lease.owner_id,
                            lease.lease_token,
                            lease.policy_fingerprint,
                        ),
                    )
                    if await cursor.fetchone() is None:
                        raise ProviderLeaseLost()
                    update(state, row["database_now"])
                    await self._write(connection, state)
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    async def record_success(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        await self._outcome(
            lease, policy, lambda state, _: _record_success_state(state, lease, policy)
        )

    async def record_failure(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        await self._outcome(
            lease,
            policy,
            lambda state, now: _record_failure_state(state, lease, now=now, policy=policy),
        )

    async def record_429(
        self, *, provider_scope: str, workload: ProviderWorkload, policy: ProviderRuntimePolicy
    ) -> None:
        del workload
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(_LOCK_STATE, (provider_scope,))
                    raw = await cursor.fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    _record_429_state(state, now=row["database_now"], policy=policy)
                    await self._write(connection, state)
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None

    async def snapshot(
        self, *, provider_scope: str, policy: ProviderRuntimePolicy
    ) -> dict[str, Any]:
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    await connection.execute(
                        _ENSURE_STATE,
                        (provider_scope, policy.fingerprint, policy.bulk_initial_slots),
                    )
                    cursor = await connection.execute(_LOCK_STATE, (provider_scope,))
                    raw = await cursor.fetchone()
                    if raw is None:
                        raise ProviderRuntimePersistenceError()
                    row = _mapping(raw)
                    state = _state_from_row(row)
                    _check_policy(state, policy)
                    cursor = await connection.execute(_DELETE_EXPIRED, (provider_scope,))
                    expired = await cursor.fetchall()
                    _advance_state(
                        state,
                        now=row["database_now"],
                        policy=policy,
                        expired_half_open=any(bool(item["half_open_probe"]) for item in expired),
                    )
                    cursor = await connection.execute(_COUNT_ACTIVE, (provider_scope,))
                    active_rows = await cursor.fetchall()
                    await self._write(connection, state)
                    counts = {
                        ProviderWorkload(str(item["workload"])): int(item["active_count"])
                        for item in active_rows
                    }
                    return _snapshot(state, counts)
        except ProviderCoordinationError:
            raise
        except Exception:
            raise ProviderRuntimePersistenceError() from None


class SyncProviderCoordinator:
    """Blocking admission facade used by synchronous graph provider clients."""

    def __init__(
        self,
        store: SyncProviderStore,
        *,
        provider_scope: str,
        policy: ProviderRuntimePolicy | None = None,
        owner_id: str | None = None,
        metrics: ProviderMetrics | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy = policy or ProviderRuntimePolicy()
        self.provider_scope = provider_scope
        self.owner_id = owner_id or f"api:{os.getpid()}:{uuid4().hex[:12]}"
        _validate_identity(self.provider_scope, self.owner_id)
        self._store = store
        self._metrics = metrics
        self._monotonic = monotonic
        self._sleep = sleep

    @classmethod
    def from_pool(cls, pool: SyncConnectionPool, **kwargs: Any) -> "SyncProviderCoordinator":
        return cls(PostgresSyncProviderStore(pool), **kwargs)

    def _inc(self, name: str, **labels: str) -> None:
        if self._metrics is not None:
            self._metrics.inc(name, **labels)

    def acquire(
        self,
        workload: ProviderWorkload | str,
        *,
        estimated_tokens: int,
        timeout_seconds: float | None = None,
    ) -> ProviderLease:
        selected = _validate_attempt(workload, estimated_tokens)
        timeout = (
            self.policy.acquire_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid provider acquisition timeout")
        started = self._monotonic()
        deadline = started + timeout
        while True:
            admission = self._store.try_acquire(
                provider_scope=self.provider_scope,
                policy=self.policy,
                owner_id=self.owner_id,
                workload=selected,
                estimated_tokens=estimated_tokens,
            )
            if admission.lease is not None:
                waited = max(0.0, self._monotonic() - started)
                if self._metrics is not None:
                    self._metrics.observe(
                        "material_graph_provider_lease_wait_seconds",
                        waited,
                        workload=selected.value,
                    )
                self._inc(
                    "material_graph_provider_admission_total",
                    workload=selected.value,
                    outcome="acquired",
                )
                return admission.lease
            if admission.outcome == "circuit_open":
                self._inc(
                    "material_graph_provider_admission_total",
                    workload=selected.value,
                    outcome="circuit_open",
                )
                raise ProviderCoordinationError(
                    "provider_coordination.circuit_open",
                    retryable=True,
                    retry_after_seconds=admission.retry_after_seconds,
                )
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                self._inc(
                    "material_graph_provider_admission_total",
                    workload=selected.value,
                    outcome="timeout",
                )
                raise ProviderCoordinationError(
                    "provider_coordination.admission_timeout", retryable=True
                )
            self._sleep(min(self.policy.poll_interval_seconds, remaining))

    def release(self, lease: ProviderLease) -> None:
        self._store.release(lease)

    @contextmanager
    def slot(
        self,
        workload: ProviderWorkload | str,
        *,
        estimated_tokens: int,
        timeout_seconds: float | None = None,
    ):
        lease = self.acquire(
            workload,
            estimated_tokens=estimated_tokens,
            timeout_seconds=timeout_seconds,
        )
        try:
            yield lease
        finally:
            self.release(lease)

    def record_success(self, lease: ProviderLease) -> None:
        self._store.record_success(lease, self.policy)
        self._inc("material_graph_provider_outcomes_total", outcome="success")

    def record_failure(self, lease: ProviderLease) -> None:
        self._store.record_failure(lease, self.policy)
        self._inc("material_graph_provider_outcomes_total", outcome="failure")

    def record_429(self, workload: ProviderWorkload | str, *, provider_wide: bool = True) -> None:
        del provider_wide
        selected = ProviderWorkload(workload)
        self._store.record_429(
            provider_scope=self.provider_scope, workload=selected, policy=self.policy
        )
        self._inc("material_graph_provider_outcomes_total", outcome="rate_limited")

    def snapshot(self) -> dict[str, Any]:
        return self._store.snapshot(provider_scope=self.provider_scope, policy=self.policy)


class AsyncProviderCoordinator:
    """Async admission facade used by background provider clients."""

    def __init__(
        self,
        store: AsyncProviderStore,
        *,
        provider_scope: str,
        policy: ProviderRuntimePolicy | None = None,
        owner_id: str | None = None,
        metrics: ProviderMetrics | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.policy = policy or ProviderRuntimePolicy()
        self.provider_scope = provider_scope
        self.owner_id = owner_id or f"worker:{os.getpid()}:{uuid4().hex[:12]}"
        _validate_identity(self.provider_scope, self.owner_id)
        self._store = store
        self._metrics = metrics
        self._monotonic = monotonic
        self._sleep = sleep

    @classmethod
    def from_pool(cls, pool: AsyncConnectionPool, **kwargs: Any) -> "AsyncProviderCoordinator":
        return cls(PostgresAsyncProviderStore(pool), **kwargs)

    def _inc(self, name: str, **labels: str) -> None:
        if self._metrics is not None:
            self._metrics.inc(name, **labels)

    async def acquire(
        self,
        workload: ProviderWorkload | str,
        *,
        estimated_tokens: int,
        timeout_seconds: float | None = None,
    ) -> ProviderLease:
        selected = _validate_attempt(workload, estimated_tokens)
        timeout = (
            self.policy.acquire_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid provider acquisition timeout")
        started = self._monotonic()
        deadline = started + timeout
        while True:
            admission = await self._store.try_acquire(
                provider_scope=self.provider_scope,
                policy=self.policy,
                owner_id=self.owner_id,
                workload=selected,
                estimated_tokens=estimated_tokens,
            )
            if admission.lease is not None:
                waited = max(0.0, self._monotonic() - started)
                if self._metrics is not None:
                    self._metrics.observe(
                        "material_graph_provider_lease_wait_seconds",
                        waited,
                        workload=selected.value,
                    )
                self._inc(
                    "material_graph_provider_admission_total",
                    workload=selected.value,
                    outcome="acquired",
                )
                return admission.lease
            if admission.outcome == "circuit_open":
                self._inc(
                    "material_graph_provider_admission_total",
                    workload=selected.value,
                    outcome="circuit_open",
                )
                raise ProviderCoordinationError(
                    "provider_coordination.circuit_open",
                    retryable=True,
                    retry_after_seconds=admission.retry_after_seconds,
                )
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                self._inc(
                    "material_graph_provider_admission_total",
                    workload=selected.value,
                    outcome="timeout",
                )
                raise ProviderCoordinationError(
                    "provider_coordination.admission_timeout", retryable=True
                )
            await self._sleep(min(self.policy.poll_interval_seconds, remaining))

    async def release(self, lease: ProviderLease) -> None:
        await self._store.release(lease)

    @asynccontextmanager
    async def slot(
        self,
        workload: ProviderWorkload | str,
        *,
        estimated_tokens: int,
        timeout_seconds: float | None = None,
    ):
        lease = await self.acquire(
            workload,
            estimated_tokens=estimated_tokens,
            timeout_seconds=timeout_seconds,
        )
        try:
            yield lease
        finally:
            await self.release(lease)

    async def record_success(self, lease: ProviderLease) -> None:
        await self._store.record_success(lease, self.policy)
        self._inc("material_graph_provider_outcomes_total", outcome="success")

    async def record_failure(self, lease: ProviderLease) -> None:
        await self._store.record_failure(lease, self.policy)
        self._inc("material_graph_provider_outcomes_total", outcome="failure")

    async def record_429(
        self, workload: ProviderWorkload | str, *, provider_wide: bool = True
    ) -> None:
        del provider_wide
        selected = ProviderWorkload(workload)
        await self._store.record_429(
            provider_scope=self.provider_scope, workload=selected, policy=self.policy
        )
        self._inc("material_graph_provider_outcomes_total", outcome="rate_limited")

    async def snapshot(self) -> dict[str, Any]:
        return await self._store.snapshot(provider_scope=self.provider_scope, policy=self.policy)


@dataclass(slots=True)
class _MemoryScope:
    state: _RuntimeState
    claims: dict[UUID, ProviderLease] = field(default_factory=dict)


class InMemoryProviderRuntime:
    """Deterministic dual sync/async store used by tests and local adapters."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._scopes: dict[str, _MemoryScope] = {}

    def sync_store(self) -> "MemorySyncProviderStore":
        return MemorySyncProviderStore(self)

    def async_store(self) -> "MemoryAsyncProviderStore":
        return MemoryAsyncProviderStore(self)

    def _scope(self, provider_scope: str, policy: ProviderRuntimePolicy) -> _MemoryScope:
        current = self._scopes.get(provider_scope)
        if current is None:
            now = self._clock()
            current = _MemoryScope(
                _RuntimeState(
                    provider_scope=provider_scope,
                    policy_fingerprint=policy.fingerprint,
                    bulk_limit=policy.bulk_initial_slots,
                    bulk_successes=0,
                    quota_window_started_at=now,
                    requests_used=0,
                    tokens_reserved=0,
                    last_429_window=None,
                    circuit_state="closed",
                    consecutive_failures=0,
                    circuit_open_until=None,
                    next_lease_token=0,
                )
            )
            self._scopes[provider_scope] = current
        _check_policy(current.state, policy)
        return current

    def _cleanup(self, scope: _MemoryScope, policy: ProviderRuntimePolicy) -> datetime:
        now = self._clock()
        expired = [claim for claim in scope.claims.values() if claim.lease_until <= now]
        for claim in expired:
            scope.claims.pop(claim.claim_id, None)
        _advance_state(
            scope.state,
            now=now,
            policy=policy,
            expired_half_open=any(claim.half_open_probe for claim in expired),
        )
        return now

    def try_acquire(self, **kwargs: Any) -> ProviderAdmission:
        provider_scope = str(kwargs["provider_scope"])
        policy = kwargs["policy"]
        owner_id = str(kwargs["owner_id"])
        workload = ProviderWorkload(kwargs["workload"])
        estimated_tokens = int(kwargs["estimated_tokens"])
        if not isinstance(policy, ProviderRuntimePolicy):
            raise TypeError("policy must be ProviderRuntimePolicy")
        with self._lock:
            scope = self._scope(provider_scope, policy)
            now = self._cleanup(scope, policy)
            counts = {
                item: sum(1 for lease in scope.claims.values() if lease.workload is item)
                for item in ProviderWorkload
            }
            decision = _admission(
                scope.state,
                now=now,
                counts=counts,
                policy=policy,
                workload=workload,
                estimated_tokens=estimated_tokens,
                half_open_active=any(lease.half_open_probe for lease in scope.claims.values()),
            )
            if decision.outcome != "ready":
                return decision
            scope.state.next_lease_token += 1
            scope.state.requests_used += 1
            scope.state.tokens_reserved += estimated_tokens
            lease = ProviderLease(
                claim_id=uuid4(),
                provider_scope=provider_scope,
                policy_fingerprint=policy.fingerprint,
                owner_id=owner_id,
                workload=workload,
                lease_token=scope.state.next_lease_token,
                estimated_tokens=estimated_tokens,
                half_open_probe=scope.state.circuit_state == "half_open",
                claimed_at=now,
                lease_until=now + timedelta(seconds=policy.lease_ttl_seconds),
            )
            scope.claims[lease.claim_id] = lease
            return ProviderAdmission("acquired", lease=lease)

    def release(self, lease: ProviderLease) -> None:
        with self._lock:
            scope = self._scopes.get(lease.provider_scope)
            if scope is not None:
                scope.claims.pop(lease.claim_id, None)

    def _claimed(
        self, lease: ProviderLease, policy: ProviderRuntimePolicy
    ) -> tuple[_MemoryScope, datetime]:
        scope = self._scope(lease.provider_scope, policy)
        now = self._cleanup(scope, policy)
        if scope.claims.get(lease.claim_id) != lease:
            raise ProviderLeaseLost()
        return scope, now

    def record_success(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        with self._lock:
            scope, _ = self._claimed(lease, policy)
            _record_success_state(scope.state, lease, policy)

    def record_failure(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        with self._lock:
            scope, now = self._claimed(lease, policy)
            _record_failure_state(scope.state, lease, now=now, policy=policy)

    def record_429(self, **kwargs: Any) -> None:
        provider_scope = str(kwargs["provider_scope"])
        policy = kwargs["policy"]
        if not isinstance(policy, ProviderRuntimePolicy):
            raise TypeError("policy must be ProviderRuntimePolicy")
        with self._lock:
            scope = self._scope(provider_scope, policy)
            now = self._cleanup(scope, policy)
            _record_429_state(scope.state, now=now, policy=policy)

    def snapshot(self, **kwargs: Any) -> dict[str, Any]:
        provider_scope = str(kwargs["provider_scope"])
        policy = kwargs["policy"]
        if not isinstance(policy, ProviderRuntimePolicy):
            raise TypeError("policy must be ProviderRuntimePolicy")
        with self._lock:
            scope = self._scope(provider_scope, policy)
            self._cleanup(scope, policy)
            counts = {
                item: sum(1 for lease in scope.claims.values() if lease.workload is item)
                for item in ProviderWorkload
            }
            return _snapshot(scope.state, counts)


class MemorySyncProviderStore:
    def __init__(self, runtime: InMemoryProviderRuntime) -> None:
        self._runtime = runtime

    def try_acquire(self, **kwargs: Any) -> ProviderAdmission:
        return self._runtime.try_acquire(**kwargs)

    def release(self, lease: ProviderLease) -> None:
        self._runtime.release(lease)

    def record_success(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        self._runtime.record_success(lease, policy)

    def record_failure(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        self._runtime.record_failure(lease, policy)

    def record_429(self, **kwargs: Any) -> None:
        self._runtime.record_429(**kwargs)

    def snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return self._runtime.snapshot(**kwargs)


class MemoryAsyncProviderStore:
    def __init__(self, runtime: InMemoryProviderRuntime) -> None:
        self._runtime = runtime

    async def try_acquire(self, **kwargs: Any) -> ProviderAdmission:
        return self._runtime.try_acquire(**kwargs)

    async def release(self, lease: ProviderLease) -> None:
        self._runtime.release(lease)

    async def record_success(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        self._runtime.record_success(lease, policy)

    async def record_failure(self, lease: ProviderLease, policy: ProviderRuntimePolicy) -> None:
        self._runtime.record_failure(lease, policy)

    async def record_429(self, **kwargs: Any) -> None:
        self._runtime.record_429(**kwargs)

    async def snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return self._runtime.snapshot(**kwargs)


__all__ = [
    "AsyncProviderCoordinator",
    "InMemoryProviderRuntime",
    "PostgresAsyncProviderStore",
    "PostgresSyncProviderStore",
    "ProviderAdmission",
    "ProviderCoordinationError",
    "ProviderLease",
    "ProviderLeaseLost",
    "ProviderRuntimePersistenceError",
    "ProviderRuntimePolicy",
    "ProviderWorkload",
    "SyncProviderCoordinator",
    "provider_scope_from_environment",
]
