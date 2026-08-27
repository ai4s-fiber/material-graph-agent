from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from material_graph.knowledge.concurrency import (
    GIB,
    AdaptiveProviderPool,
    AdmissionAction,
    AdmissionSnapshot,
    GlobalAdmissionController,
    LLMWorkload,
    PoolAcquireTimeout,
    PoolQueueFull,
    ProviderFailure,
    ProviderPoolPolicy,
    QueuePressure,
    SharedLLMLimiter,
    SharedLLMLimiterPolicy,
)


def _pool_policy(**overrides: object) -> ProviderPoolPolicy:
    values: dict[str, object] = {
        "pool_id": "embedding",
        "initial": 2,
        "hard_max": 8,
        "success_window": 3,
        "latency_target_seconds": 1.0,
        "retry_max_attempts": 4,
        "backoff_base_seconds": 2.0,
        "backoff_max_seconds": 10.0,
        "max_waiters": 2,
        "acquire_timeout_seconds": 0.05,
    }
    values.update(overrides)
    return ProviderPoolPolicy.model_validate(values)


def test_provider_policy_rejects_invalid_capacity_and_backoff() -> None:
    with pytest.raises(ValidationError):
        _pool_policy(initial=5, hard_max=4)
    with pytest.raises(ValidationError):
        _pool_policy(backoff_base_seconds=20, backoff_max_seconds=10)


def test_low_latency_windows_grow_only_to_effective_max() -> None:
    async def run() -> None:
        pool = AdaptiveProviderPool(
            _pool_policy(),
            provider_advertised_limit=6,
            resource_limit=4,
        )

        for _ in range(3):
            await pool.record_success(latency_seconds=0.2)
        assert pool.current_limit == 3

        for _ in range(3):
            await pool.record_success(latency_seconds=0.2)
        assert pool.current_limit == 4

        for _ in range(6):
            await pool.record_success(latency_seconds=0.2)
        assert pool.current_limit == 4
        assert pool.effective_max == 4

        await pool.record_success(latency_seconds=2.0)
        for _ in range(2):
            await pool.record_success(latency_seconds=0.2)
        assert pool.current_limit == 4

        await pool.update_effective_max(resource_limit=3)
        assert pool.current_limit == 3
        assert pool.effective_max == 3

    asyncio.run(run())


def test_global_pressure_prevents_growth_without_reducing_other_pools() -> None:
    async def run() -> None:
        embedding = AdaptiveProviderPool(_pool_policy(pool_id="embedding"))
        mineru = AdaptiveProviderPool(_pool_policy(pool_id="mineru_submit", initial=4))
        decision = GlobalAdmissionController().decide(
            AdmissionSnapshot(spool_used_bytes=70, spool_max_bytes=100)
        )

        for _ in range(3):
            await embedding.record_success(
                latency_seconds=0.1,
                growth_allowed=decision.allow_growth,
            )

        assert decision.action is AdmissionAction.STOP_GROWTH
        assert embedding.current_limit == 2
        assert mineru.current_limit == 4

    asyncio.run(run())


def test_429_honors_retry_after_and_halves_only_that_pool() -> None:
    async def run() -> None:
        mineru = AdaptiveProviderPool(_pool_policy(pool_id="mineru_submit", initial=8))
        embedding = AdaptiveProviderPool(_pool_policy(pool_id="embedding", initial=6))

        directive = await mineru.record_rate_limited(
            retry_after_seconds=7,
            attempt=1,
        )

        assert directive.retryable is True
        assert directive.delay_seconds == 7
        assert directive.reason == "rate_limited"
        assert mineru.current_limit == 4
        assert embedding.current_limit == 6

    asyncio.run(run())


def test_retryable_failures_use_bounded_full_jitter_and_auth_is_not_retried() -> None:
    async def run() -> None:
        pool = AdaptiveProviderPool(
            _pool_policy(initial=8),
            random_fn=lambda: 0.5,
        )

        service_error = await pool.record_failure(
            ProviderFailure.HTTP_503,
            attempt=2,
        )
        assert service_error.retryable is True
        assert service_error.delay_seconds == 2.0
        assert pool.current_limit == 4

        timeout = await pool.record_failure(ProviderFailure.TIMEOUT, attempt=3)
        assert timeout.retryable is True
        assert timeout.delay_seconds == 4.0
        assert pool.current_limit == 2

        authentication = await pool.record_failure(
            ProviderFailure.AUTHENTICATION,
            attempt=1,
        )
        schema = await pool.record_failure(ProviderFailure.SCHEMA, attempt=1)
        assert authentication.retryable is False
        assert schema.retryable is False
        assert authentication.delay_seconds == 0
        assert pool.current_limit == 2

        exhausted = await pool.record_failure(ProviderFailure.HTTP_500, attempt=4)
        assert exhausted.retryable is False
        assert exhausted.delay_seconds == 0
        assert pool.current_limit == 1

    asyncio.run(run())


def test_provider_wait_queue_is_bounded_and_slots_release_cleanly() -> None:
    async def run() -> None:
        pool = AdaptiveProviderPool(
            _pool_policy(initial=1, hard_max=1, max_waiters=1),
        )
        first = await pool.acquire()
        waiting = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        assert pool.waiter_count == 1

        with pytest.raises(PoolQueueFull):
            await pool.acquire()

        await first.release()
        second = await waiting
        assert pool.active_count == 1
        await second.release()
        assert pool.active_count == 0

    asyncio.run(run())


def test_provider_acquire_timeout_is_explicit() -> None:
    async def run() -> None:
        pool = AdaptiveProviderPool(
            _pool_policy(
                initial=1,
                hard_max=1,
                max_waiters=1,
                acquire_timeout_seconds=0.001,
            )
        )
        lease = await pool.acquire()
        with pytest.raises(PoolAcquireTimeout):
            await pool.acquire()
        await lease.release()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("snapshot", "action", "reason"),
    [
        (
            AdmissionSnapshot(spool_used_bytes=69, spool_max_bytes=100),
            AdmissionAction.ALLOW,
            None,
        ),
        (
            AdmissionSnapshot(spool_used_bytes=70, spool_max_bytes=100),
            AdmissionAction.STOP_GROWTH,
            "spool_growth_threshold",
        ),
        (
            AdmissionSnapshot(spool_used_bytes=85, spool_max_bytes=100),
            AdmissionAction.PAUSE_BODY,
            "spool_pause_threshold",
        ),
        (
            AdmissionSnapshot(filesystem_used_ratio=0.75),
            AdmissionAction.PAUSE_BODY,
            "filesystem_alert_threshold",
        ),
        (
            AdmissionSnapshot(filesystem_used_ratio=0.80),
            AdmissionAction.HARD_STOP,
            "filesystem_stop_threshold",
        ),
        (
            AdmissionSnapshot(free_bytes=31 * GIB),
            AdmissionAction.PAUSE_BODY,
            "free_space_alert_threshold",
        ),
        (
            AdmissionSnapshot(free_bytes=25 * GIB),
            AdmissionAction.HARD_STOP,
            "free_space_stop_threshold",
        ),
        (
            AdmissionSnapshot(derived_bytes=75 * GIB),
            AdmissionAction.PAUSE_BODY,
            "derived_target_threshold",
        ),
        (
            AdmissionSnapshot(derived_bytes=80 * GIB),
            AdmissionAction.HARD_STOP,
            "derived_hard_cap",
        ),
        (
            AdmissionSnapshot(queues={"parse": QueuePressure(depth=100, capacity=100)}),
            AdmissionAction.PAUSE_BODY,
            "queue_full:parse",
        ),
    ],
)
def test_global_admission_thresholds_are_explicit(
    snapshot: AdmissionSnapshot,
    action: AdmissionAction,
    reason: str | None,
) -> None:
    decision = GlobalAdmissionController().decide(snapshot)

    assert decision.action is action
    assert decision.allow_growth is (action is AdmissionAction.ALLOW)
    assert decision.allow_body is (
        action
        in {
            AdmissionAction.ALLOW,
            AdmissionAction.STOP_GROWTH,
        }
    )
    assert decision.allow_metadata is not (action is AdmissionAction.HARD_STOP)
    if reason is not None:
        assert reason in decision.reasons


def test_memory_pressure_and_full_spool_pause_or_stop_admission() -> None:
    controller = GlobalAdmissionController()

    pressure = controller.decide(AdmissionSnapshot(memory_pressure=True))
    full = controller.decide(AdmissionSnapshot(spool_used_bytes=100, spool_max_bytes=100))

    assert pressure.action is AdmissionAction.PAUSE_BODY
    assert "memory_pressure" in pressure.reasons
    assert full.action is AdmissionAction.HARD_STOP
    assert "spool_capacity_reached" in full.reasons


def test_llm_bulk_cannot_consume_two_reserved_interactive_slots() -> None:
    async def run() -> None:
        limiter = SharedLLMLimiter(
            SharedLLMLimiterPolicy(
                total_limit=4,
                reserved_interactive=2,
                max_waiters=4,
                acquire_timeout_seconds=0.05,
            )
        )
        bulk_one = await limiter.acquire(LLMWorkload.BULK)
        bulk_two = await limiter.acquire(LLMWorkload.BULK)
        blocked_bulk = asyncio.create_task(limiter.acquire(LLMWorkload.BULK))
        await asyncio.sleep(0)

        interactive_one = await limiter.acquire(LLMWorkload.INTERACTIVE)
        interactive_two = await limiter.acquire(LLMWorkload.INTERACTIVE)

        assert limiter.active_bulk == 2
        assert limiter.active_interactive == 2
        assert blocked_bulk.done() is False

        blocked_bulk.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked_bulk
        for lease in (bulk_one, bulk_two, interactive_one, interactive_two):
            await lease.release()

    asyncio.run(run())


def test_waiting_interactive_request_has_priority_over_waiting_bulk() -> None:
    async def run() -> None:
        limiter = SharedLLMLimiter(
            SharedLLMLimiterPolicy(
                total_limit=4,
                reserved_interactive=2,
                max_waiters=4,
                acquire_timeout_seconds=0.1,
            )
        )
        bulk_one = await limiter.acquire(LLMWorkload.BULK)
        bulk_two = await limiter.acquire(LLMWorkload.BULK)
        interactive_one = await limiter.acquire(LLMWorkload.INTERACTIVE)
        interactive_two = await limiter.acquire(LLMWorkload.INTERACTIVE)

        bulk_waiter = asyncio.create_task(limiter.acquire(LLMWorkload.BULK))
        await asyncio.sleep(0)
        interactive_waiter = asyncio.create_task(limiter.acquire(LLMWorkload.INTERACTIVE))
        await asyncio.sleep(0)
        assert limiter.waiting_interactive == 1
        assert limiter.waiting_bulk == 1

        await interactive_one.release()
        next_interactive = await asyncio.wait_for(interactive_waiter, timeout=0.05)
        assert next_interactive.workload is LLMWorkload.INTERACTIVE
        assert bulk_waiter.done() is False

        await bulk_one.release()
        next_bulk = await asyncio.wait_for(bulk_waiter, timeout=0.05)
        assert next_bulk.workload is LLMWorkload.BULK

        for lease in (
            bulk_two,
            interactive_two,
            next_interactive,
            next_bulk,
        ):
            await lease.release()

    asyncio.run(run())


def test_shared_llm_wait_queue_is_bounded() -> None:
    async def run() -> None:
        limiter = SharedLLMLimiter(
            SharedLLMLimiterPolicy(
                total_limit=4,
                reserved_interactive=2,
                max_waiters=1,
                acquire_timeout_seconds=0.05,
            )
        )
        leases = [
            await limiter.acquire(LLMWorkload.BULK),
            await limiter.acquire(LLMWorkload.BULK),
            await limiter.acquire(LLMWorkload.INTERACTIVE),
            await limiter.acquire(LLMWorkload.INTERACTIVE),
        ]
        waiting = asyncio.create_task(limiter.acquire(LLMWorkload.INTERACTIVE))
        await asyncio.sleep(0)

        with pytest.raises(PoolQueueFull):
            await limiter.acquire(LLMWorkload.BULK)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        for lease in leases:
            await lease.release()

    asyncio.run(run())


def test_shared_llm_policy_always_reserves_at_least_two_slots() -> None:
    with pytest.raises(ValidationError):
        SharedLLMLimiterPolicy(total_limit=4, reserved_interactive=1)
    with pytest.raises(ValidationError):
        SharedLLMLimiterPolicy(total_limit=2, reserved_interactive=2)
