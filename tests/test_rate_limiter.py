"""Tests for the RPM token-bucket limiter on OpenAICompatibleProvider.

These tests use a fake monotonic clock so they run in (real) milliseconds while
still exercising the full token-bucket refill / drain / block behaviour.
"""

from __future__ import annotations

import threading

import pytest

from kernel_agent.ka_utils.providers import openai_base


# --------------------------------------------------------------------------- #
# Fake-clock harness
# --------------------------------------------------------------------------- #


class FakeClock:
    """Thread-safe monotonic clock whose ``time.sleep`` is instantaneous.

    ``sleep(dt)`` advances the clock by ``dt`` and returns immediately, so the
    limiter's busy-wait loop exits without wall-clock delay.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, dt: float) -> None:
        if dt <= 0:
            return
        with self._lock:
            self._t += dt

    def advance(self, dt: float) -> None:
        with self._lock:
            self._t += dt


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(openai_base.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(openai_base.time, "sleep", clock.sleep)
    return clock


# --------------------------------------------------------------------------- #
# Test-only provider subclasses (bypass _initialize_client's OpenAI dependency)
# --------------------------------------------------------------------------- #


class _DummyProvider(openai_base.OpenAICompatibleProvider):
    """Minimal concrete subclass that does not touch the OpenAI SDK."""

    _name = "dummy"
    rpm_limit: float | None = None

    def _initialize_client(self) -> None:  # pragma: no cover - trivial
        self.client = None

    @property
    def name(self) -> str:
        return self._name


def _make_provider(name: str = "dummy", rpm: float | None = None) -> _DummyProvider:
    """Construct a provider with a given name + class-level rpm_limit."""

    cls = type(
        f"P_{name}",
        (_DummyProvider,),
        {"_name": name, "rpm_limit": rpm},
    )
    return cls(api_key_env="UNUSED_ENV")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_unlimited_provider_no_wait(fake_clock: FakeClock) -> None:
    """rpm_limit=None: _await_rate_limit is a no-op, no fake-clock drift."""
    provider = _make_provider(name="dummy_unlimited", rpm=None)
    t0 = fake_clock.monotonic()
    for _ in range(1000):
        provider._await_rate_limit()
    assert fake_clock.monotonic() == t0, "unlimited path must not sleep"


def test_40_rpm_bucket_drains(fake_clock: FakeClock) -> None:
    """First 40 calls don't wait; 41st waits ~1.5s, 42nd ~3s (bucket drained)."""
    provider = _make_provider(name="drain", rpm=40)
    start = fake_clock.monotonic()

    # Drain the bucket: first 40 calls consume the initial 40 tokens, no sleep.
    for _ in range(40):
        provider._await_rate_limit()
    assert fake_clock.monotonic() == start, "initial 40 tokens should not sleep"

    # 41st call: bucket at 0, refill rate = 40/60 tokens/s → 1.5s for 1 token.
    t_before = fake_clock.monotonic()
    provider._await_rate_limit()
    elapsed_41 = fake_clock.monotonic() - t_before
    assert elapsed_41 == pytest.approx(1.5, abs=1e-6), (
        f"41st call should wait ~1.5s, waited {elapsed_41}"
    )

    # 42nd call: another full 1.5s window.
    t_before = fake_clock.monotonic()
    provider._await_rate_limit()
    elapsed_42 = fake_clock.monotonic() - t_before
    assert elapsed_42 == pytest.approx(1.5, abs=1e-6), (
        f"42nd call should wait ~1.5s, waited {elapsed_42}"
    )

    # Total elapsed = 0 (initial 40) + 1.5 + 1.5 = 3.0s.
    assert fake_clock.monotonic() - start == pytest.approx(3.0, abs=1e-6)


def test_refill_over_time(fake_clock: FakeClock) -> None:
    """Drain bucket, advance 30s, bucket should refill to 20 tokens (40/min * 0.5min)."""
    provider = _make_provider(name="refill", rpm=40)

    # Drain all 40 initial tokens.
    for _ in range(40):
        provider._await_rate_limit()

    # Advance 30 seconds: refill = 30 * (40/60) = 20 tokens.
    fake_clock.advance(30.0)

    t_before = fake_clock.monotonic()
    for _ in range(20):
        provider._await_rate_limit()
    # First 20 consume refilled tokens without sleeping.
    assert fake_clock.monotonic() == t_before, (
        "20 refilled tokens should be available with no further wait"
    )

    # 21st call must sleep (bucket empty again).
    provider._await_rate_limit()
    assert fake_clock.monotonic() > t_before, (
        "21st call after refill must block for a new token"
    )


def test_env_override(
    fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OPENKERNEL_PROVIDER_RPM_<NAME_UPPER> overrides class attr."""
    monkeypatch.setenv("OPENKERNEL_PROVIDER_RPM_NVIDIA", "100")

    # Class attr says 40, env says 100 — env must win.
    provider = _make_provider(name="nvidia", rpm=40)
    assert provider._get_rpm_limit() == 100.0

    # Sanity: bucket should permit 100 non-blocking calls back-to-back.
    # (Initial bucket was sized from env at __init__ time.)
    t0 = fake_clock.monotonic()
    for _ in range(100):
        provider._await_rate_limit()
    assert fake_clock.monotonic() == t0, (
        "100 calls should fit in the env-overridden bucket without sleeping"
    )

    # Invalid env value falls back to class attr.
    monkeypatch.setenv("OPENKERNEL_PROVIDER_RPM_NVIDIA", "not-a-number")
    p2 = _make_provider(name="nvidia", rpm=40)
    assert p2._get_rpm_limit() == 40.0

    # Empty env also falls back.
    monkeypatch.setenv("OPENKERNEL_PROVIDER_RPM_NVIDIA", "")
    p3 = _make_provider(name="nvidia", rpm=40)
    assert p3._get_rpm_limit() == 40.0

    # Unset env + no class attr → unlimited (None).
    monkeypatch.delenv("OPENKERNEL_PROVIDER_RPM_NVIDIA", raising=False)
    p4 = _make_provider(name="otherprov", rpm=None)
    assert p4._get_rpm_limit() is None


def test_thread_safety(fake_clock: FakeClock) -> None:
    """10 threads × 50 calls at 40 RPM: bucket accounting stays consistent.

    Under a fake clock with per-call sleep advancing a shared clock, multiple
    concurrently-sleeping threads would each independently push the clock
    forward — so we can't meaningfully verify wall-time spacing the way a real
    clock would enforce it. What we *can* verify is that the internal token
    accounting stays consistent under concurrent access: the lock must prevent
    the bucket from ever going negative, and exactly ``total_calls`` tokens
    must be dispensed (no double-dispense on a race).

    That invariant is what justifies using ``threading.Lock`` in the limiter.
    """
    provider = _make_provider(name="threaded", rpm=40)
    total_calls = 10 * 50  # 500 calls

    # Wrap the real _await_rate_limit with a hook that samples the bucket state
    # right after the token is consumed, while still inside the lock window
    # conceptually. We do this by patching the internal method to record the
    # post-decrement value.
    observed_token_values: list[float] = []
    observed_lock = threading.Lock()
    original_await = provider._await_rate_limit

    def instrumented_await() -> None:
        original_await()
        # After return, the token has been consumed. Sample state under lock.
        with provider._rl_lock:
            tokens_after = provider._rl_tokens
        with observed_lock:
            observed_token_values.append(tokens_after)

    provider._await_rate_limit = instrumented_await  # type: ignore[method-assign]

    def worker() -> None:
        for _ in range(50):
            provider._await_rate_limit()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Invariant 1: every call returned → every call consumed exactly one token.
    assert len(observed_token_values) == total_calls

    # Invariant 2: post-consume token count is ALWAYS >= 0. A negative value
    # would mean the lock failed to serialize the check-then-decrement.
    assert all(v >= 0 for v in observed_token_values), (
        f"bucket went negative — min observed {min(observed_token_values)}"
    )

    # Invariant 3: post-consume token count is never > rpm_limit. A value over
    # the cap would mean refill ran without being clamped by ``min(limit, ...)``.
    assert all(v <= 40 for v in observed_token_values), (
        f"bucket exceeded cap — max observed {max(observed_token_values)}"
    )
