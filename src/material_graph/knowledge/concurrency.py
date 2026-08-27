"""Provider-isolated adaptive concurrency and global ingestion backpressure."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Callable
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator


GIB = 1024**3


class PoolQueueFull(RuntimeError):
    """Raised before an unbounded provider or shared-LLM wait queue can form."""


class PoolAcquireTimeout(TimeoutError):
    """Raised when a caller cannot obtain capacity inside its wait deadline."""


class ProviderFailure(StrEnum):
    HTTP_500 = "http_500"
    HTTP_502 = "http_502"
    HTTP_503 = "http_503"
    HTTP_504 = "http_504"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    SCHEMA = "schema"


_RETRYABLE_FAILURES = frozenset(
    {
        ProviderFailure.HTTP_500,
        ProviderFailure.HTTP_502,
        ProviderFailure.HTTP_503,
        ProviderFailure.HTTP_504,
        ProviderFailure.TIMEOUT,
    }
)


class ProviderPoolPolicy(BaseModel):
    """Validated AIMD, retry, and bounded-wait settings for one provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    initial: int = Field(ge=1)
    hard_max: int = Field(ge=1)
    success_window: int = Field(default=20, ge=1)
    latency_target_seconds: float = Field(gt=0)
    retry_max_attempts: int = Field(default=6, ge=1, le=8)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=60.0, gt=0)
    max_waiters: int = Field(default=100, ge=0)
    acquire_timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_limits(self) -> "ProviderPoolPolicy":
        if self.initial > self.hard_max:
            raise ValueError("initial concurrency cannot exceed hard_max")
        if self.backoff_base_seconds > self.backoff_max_seconds:
            raise ValueError("backoff base cannot exceed maximum backoff")
        return self


class RetryDirective(BaseModel):
    """Persistable retry decision; callers own checkpointing and sleeping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retryable: bool
    delay_seconds: float = Field(ge=0)
    reason: str = Field(min_length=1)
    attempt: int = Field(ge=1)


def _validated_external_limit(value: int | None, *, name: str, fallback: int) -> int:
    if value is None:
        return fallback
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


class ProviderLease:
    """Idempotently releases one adaptive-provider slot."""

    def __init__(self, pool: "AdaptiveProviderPool") -> None:
        self._pool = pool
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._pool._release()

    async def __aenter__(self) -> "ProviderLease":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()


class AdaptiveProviderPool:
    """One provider's isolated dynamic limit, wait queue, and retry feedback."""

    def __init__(
        self,
        policy: ProviderPoolPolicy,
        *,
        provider_advertised_limit: int | None = None,
        resource_limit: int | None = None,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self.policy = policy
        self._provider_limit = _validated_external_limit(
            provider_advertised_limit,
            name="provider_advertised_limit",
            fallback=policy.hard_max,
        )
        self._resource_limit = _validated_external_limit(
            resource_limit,
            name="resource_limit",
            fallback=policy.hard_max,
        )
        self._current_limit = min(policy.initial, self.effective_max)
        self._active_count = 0
        self._waiter_count = 0
        self._healthy_successes = 0
        self._condition = asyncio.Condition()
        self._random_fn = random_fn

    @property
    def current_limit(self) -> int:
        return self._current_limit

    @property
    def effective_max(self) -> int:
        return min(
            self.policy.hard_max,
            self._provider_limit,
            self._resource_limit,
        )

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def waiter_count(self) -> int:
        return self._waiter_count

    async def update_effective_max(
        self,
        *,
        provider_advertised_limit: int | None = None,
        resource_limit: int | None = None,
    ) -> None:
        provider_limit = _validated_external_limit(
            provider_advertised_limit,
            name="provider_advertised_limit",
            fallback=self.policy.hard_max,
        )
        resource = _validated_external_limit(
            resource_limit,
            name="resource_limit",
            fallback=self.policy.hard_max,
        )
        async with self._condition:
            self._provider_limit = provider_limit
            self._resource_limit = resource
            self._current_limit = min(self._current_limit, self.effective_max)
            self._healthy_successes = 0
            self._condition.notify_all()

    async def acquire(self, *, timeout_seconds: float | None = None) -> ProviderLease:
        timeout = (
            self.policy.acquire_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if timeout <= 0:
            raise PoolAcquireTimeout(f"provider pool {self.policy.pool_id} wait timed out")

        async with self._condition:
            if self._active_count < self._current_limit:
                self._active_count += 1
                return ProviderLease(self)
            if self._waiter_count >= self.policy.max_waiters:
                raise PoolQueueFull(f"provider pool {self.policy.pool_id} wait queue is full")

            self._waiter_count += 1
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._active_count < self._current_limit),
                    timeout=timeout,
                )
                self._active_count += 1
                return ProviderLease(self)
            except TimeoutError as error:
                raise PoolAcquireTimeout(
                    f"provider pool {self.policy.pool_id} wait timed out"
                ) from error
            finally:
                self._waiter_count -= 1

    async def slot(self, *, timeout_seconds: float | None = None) -> ProviderLease:
        """Return an async-context-manager lease for one provider call."""

        return await self.acquire(timeout_seconds=timeout_seconds)

    async def _release(self) -> None:
        async with self._condition:
            if self._active_count <= 0:
                raise RuntimeError("provider pool release without an active lease")
            self._active_count -= 1
            self._condition.notify_all()

    async def record_success(
        self,
        *,
        latency_seconds: float,
        growth_allowed: bool = True,
    ) -> bool:
        if latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")
        async with self._condition:
            healthy = latency_seconds <= self.policy.latency_target_seconds
            if not healthy or not growth_allowed:
                self._healthy_successes = 0
                return False
            self._healthy_successes += 1
            if self._healthy_successes < self.policy.success_window:
                return False
            self._healthy_successes = 0
            new_limit = min(self._current_limit + 1, self.effective_max)
            grew = new_limit > self._current_limit
            self._current_limit = new_limit
            if grew:
                self._condition.notify_all()
            return grew

    async def _halve(self) -> None:
        async with self._condition:
            self._current_limit = max(1, self._current_limit // 2)
            self._healthy_successes = 0
            self._condition.notify_all()

    def _full_jitter_delay(self, attempt: int) -> float:
        exponent = max(0, attempt - 1)
        ceiling = min(
            self.policy.backoff_max_seconds,
            self.policy.backoff_base_seconds * (2**exponent),
        )
        sample = float(self._random_fn())
        if not math.isfinite(sample):
            sample = 1.0
        sample = min(1.0, max(0.0, sample))
        return ceiling * sample

    async def record_rate_limited(
        self,
        *,
        retry_after_seconds: float | None,
        attempt: int,
    ) -> RetryDirective:
        if attempt < 1:
            raise ValueError("attempt must be at least one")
        await self._halve()
        retryable = attempt < self.policy.retry_max_attempts
        if not retryable:
            delay = 0.0
        elif retry_after_seconds is None or not math.isfinite(retry_after_seconds):
            delay = self._full_jitter_delay(attempt)
        else:
            delay = min(
                self.policy.backoff_max_seconds,
                max(0.0, retry_after_seconds),
            )
        return RetryDirective(
            retryable=retryable,
            delay_seconds=delay,
            reason="rate_limited",
            attempt=attempt,
        )

    async def record_failure(
        self,
        failure: ProviderFailure,
        *,
        attempt: int,
    ) -> RetryDirective:
        if attempt < 1:
            raise ValueError("attempt must be at least one")
        if failure not in _RETRYABLE_FAILURES:
            return RetryDirective(
                retryable=False,
                delay_seconds=0,
                reason=failure.value,
                attempt=attempt,
            )

        await self._halve()
        retryable = attempt < self.policy.retry_max_attempts
        return RetryDirective(
            retryable=retryable,
            delay_seconds=(self._full_jitter_delay(attempt) if retryable else 0),
            reason=failure.value,
            attempt=attempt,
        )


class AdmissionAction(StrEnum):
    ALLOW = "allow"
    STOP_GROWTH = "stop_growth"
    PAUSE_BODY = "pause_body"
    HARD_STOP = "hard_stop"


class QueuePressure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    depth: int = Field(ge=0)
    capacity: int = Field(ge=1)

    @property
    def full(self) -> bool:
        return self.depth >= self.capacity


class AdmissionSnapshot(BaseModel):
    """One side-effect-free capacity snapshot used before source body reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spool_used_bytes: int = Field(default=0, ge=0)
    spool_max_bytes: int = Field(default=8 * GIB, ge=1)
    filesystem_used_ratio: float = Field(default=0.0, ge=0, le=1)
    free_bytes: int = Field(default=130 * GIB, ge=0)
    derived_bytes: int = Field(default=0, ge=0)
    queues: dict[str, QueuePressure] = Field(default_factory=dict)
    memory_pressure: bool = False

    @property
    def spool_ratio(self) -> float:
        return self.spool_used_bytes / self.spool_max_bytes


class GlobalAdmissionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spool_growth_ratio: float = Field(default=0.70, ge=0, le=1)
    spool_pause_ratio: float = Field(default=0.85, ge=0, le=1)
    filesystem_alert_ratio: float = Field(default=0.75, ge=0, le=1)
    filesystem_stop_ratio: float = Field(default=0.80, ge=0, le=1)
    free_space_alert_bytes: int = Field(default=32 * GIB, ge=0)
    free_space_stop_bytes: int = Field(default=26 * GIB, ge=0)
    derived_target_bytes: int = Field(default=75 * GIB, ge=1)
    derived_hard_cap_bytes: int = Field(default=80 * GIB, ge=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "GlobalAdmissionPolicy":
        if self.spool_growth_ratio > self.spool_pause_ratio:
            raise ValueError("spool growth threshold cannot exceed pause threshold")
        if self.filesystem_alert_ratio > self.filesystem_stop_ratio:
            raise ValueError("filesystem alert threshold cannot exceed stop threshold")
        if self.free_space_stop_bytes > self.free_space_alert_bytes:
            raise ValueError("free-space stop threshold cannot exceed alert threshold")
        if self.derived_target_bytes > self.derived_hard_cap_bytes:
            raise ValueError("derived target cannot exceed hard cap")
        return self


class AdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: AdmissionAction
    allow_growth: bool
    allow_body: bool
    allow_metadata: bool
    reasons: list[str] = Field(default_factory=list)


class GlobalAdmissionController:
    """Convert global resource pressure into an explicit ingestion decision."""

    def __init__(self, policy: GlobalAdmissionPolicy | None = None) -> None:
        self.policy = policy or GlobalAdmissionPolicy()

    def decide(self, snapshot: AdmissionSnapshot) -> AdmissionDecision:
        reasons: list[str] = []
        stop_growth = False
        pause_body = False
        hard_stop = False

        if snapshot.spool_ratio >= self.policy.spool_growth_ratio:
            stop_growth = True
            reasons.append("spool_growth_threshold")
        if snapshot.spool_ratio >= self.policy.spool_pause_ratio:
            pause_body = True
            reasons.append("spool_pause_threshold")
        if snapshot.spool_ratio >= 1.0:
            hard_stop = True
            reasons.append("spool_capacity_reached")

        if snapshot.filesystem_used_ratio >= self.policy.filesystem_alert_ratio:
            pause_body = True
            reasons.append("filesystem_alert_threshold")
        if snapshot.filesystem_used_ratio >= self.policy.filesystem_stop_ratio:
            hard_stop = True
            reasons.append("filesystem_stop_threshold")

        if snapshot.free_bytes < self.policy.free_space_alert_bytes:
            pause_body = True
            reasons.append("free_space_alert_threshold")
        if snapshot.free_bytes < self.policy.free_space_stop_bytes:
            hard_stop = True
            reasons.append("free_space_stop_threshold")

        if snapshot.derived_bytes >= self.policy.derived_target_bytes:
            pause_body = True
            reasons.append("derived_target_threshold")
        if snapshot.derived_bytes >= self.policy.derived_hard_cap_bytes:
            hard_stop = True
            reasons.append("derived_hard_cap")

        for queue_id, pressure in sorted(snapshot.queues.items()):
            if pressure.full:
                pause_body = True
                reasons.append(f"queue_full:{queue_id}")
        if snapshot.memory_pressure:
            pause_body = True
            reasons.append("memory_pressure")

        if hard_stop:
            action = AdmissionAction.HARD_STOP
        elif pause_body:
            action = AdmissionAction.PAUSE_BODY
        elif stop_growth:
            action = AdmissionAction.STOP_GROWTH
        else:
            action = AdmissionAction.ALLOW
        return AdmissionDecision(
            action=action,
            allow_growth=action is AdmissionAction.ALLOW,
            allow_body=action in {AdmissionAction.ALLOW, AdmissionAction.STOP_GROWTH},
            allow_metadata=action is not AdmissionAction.HARD_STOP,
            reasons=reasons,
        )


class LLMWorkload(StrEnum):
    INTERACTIVE = "llm_interactive"
    BULK = "llm_bulk_extract"


class SharedLLMLimiterPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_limit: int = Field(default=16, ge=3)
    reserved_interactive: int = Field(default=2, ge=2)
    max_waiters: int = Field(default=200, ge=0)
    acquire_timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def preserve_bulk_and_interactive_capacity(self) -> "SharedLLMLimiterPolicy":
        if self.reserved_interactive >= self.total_limit:
            raise ValueError("reserved interactive slots must be below total_limit")
        return self


class SharedLLMLease:
    def __init__(self, limiter: "SharedLLMLimiter", workload: LLMWorkload) -> None:
        self._limiter = limiter
        self.workload = workload
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._limiter._release(self.workload)

    async def __aenter__(self) -> "SharedLLMLease":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()


class SharedLLMLimiter:
    """Shared provider capacity with two protected interactive slots."""

    def __init__(self, policy: SharedLLMLimiterPolicy | None = None) -> None:
        self.policy = policy or SharedLLMLimiterPolicy()
        self._condition = asyncio.Condition()
        self._active_interactive = 0
        self._active_bulk = 0
        self._waiting_interactive = 0
        self._waiting_bulk = 0

    @property
    def active_interactive(self) -> int:
        return self._active_interactive

    @property
    def active_bulk(self) -> int:
        return self._active_bulk

    @property
    def waiting_interactive(self) -> int:
        return self._waiting_interactive

    @property
    def waiting_bulk(self) -> int:
        return self._waiting_bulk

    @property
    def active_total(self) -> int:
        return self._active_interactive + self._active_bulk

    @property
    def bulk_capacity(self) -> int:
        return self.policy.total_limit - self.policy.reserved_interactive

    def _can_acquire(self, workload: LLMWorkload) -> bool:
        if self.active_total >= self.policy.total_limit:
            return False
        if workload is LLMWorkload.INTERACTIVE:
            return True
        return self._waiting_interactive == 0 and self._active_bulk < self.bulk_capacity

    async def acquire(
        self,
        workload: LLMWorkload,
        *,
        timeout_seconds: float | None = None,
    ) -> SharedLLMLease:
        timeout = (
            self.policy.acquire_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if timeout <= 0:
            raise PoolAcquireTimeout("shared LLM limiter wait timed out")

        async with self._condition:
            if self._can_acquire(workload):
                self._increment_active(workload)
                return SharedLLMLease(self, workload)
            if self._waiting_interactive + self._waiting_bulk >= self.policy.max_waiters:
                raise PoolQueueFull("shared LLM limiter wait queue is full")

            self._increment_waiting(workload)
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._can_acquire(workload)),
                    timeout=timeout,
                )
                self._increment_active(workload)
                return SharedLLMLease(self, workload)
            except TimeoutError as error:
                raise PoolAcquireTimeout("shared LLM limiter wait timed out") from error
            finally:
                self._decrement_waiting(workload)

    def _increment_active(self, workload: LLMWorkload) -> None:
        if workload is LLMWorkload.INTERACTIVE:
            self._active_interactive += 1
        else:
            self._active_bulk += 1

    def _increment_waiting(self, workload: LLMWorkload) -> None:
        if workload is LLMWorkload.INTERACTIVE:
            self._waiting_interactive += 1
        else:
            self._waiting_bulk += 1

    def _decrement_waiting(self, workload: LLMWorkload) -> None:
        if workload is LLMWorkload.INTERACTIVE:
            self._waiting_interactive -= 1
        else:
            self._waiting_bulk -= 1

    async def _release(self, workload: LLMWorkload) -> None:
        async with self._condition:
            if workload is LLMWorkload.INTERACTIVE:
                if self._active_interactive <= 0:
                    raise RuntimeError("interactive LLM release without an active lease")
                self._active_interactive -= 1
            else:
                if self._active_bulk <= 0:
                    raise RuntimeError("bulk LLM release without an active lease")
                self._active_bulk -= 1
            self._condition.notify_all()


__all__ = [
    "GIB",
    "AdaptiveProviderPool",
    "AdmissionAction",
    "AdmissionDecision",
    "AdmissionSnapshot",
    "GlobalAdmissionController",
    "GlobalAdmissionPolicy",
    "LLMWorkload",
    "PoolAcquireTimeout",
    "PoolQueueFull",
    "ProviderFailure",
    "ProviderLease",
    "ProviderPoolPolicy",
    "QueuePressure",
    "RetryDirective",
    "SharedLLMLease",
    "SharedLLMLimiter",
    "SharedLLMLimiterPolicy",
]
