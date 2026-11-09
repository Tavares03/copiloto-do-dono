from datetime import UTC, datetime

from copiloto.llm.resilience import CircuitBreaker, retry_after_seconds
from fakes import FakeClock

NOW = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)


def test_retry_after_in_seconds():
    assert retry_after_seconds({"retry-after": "7"}) == 7.0
    assert retry_after_seconds({"retry-after": "1.5"}) == 1.5


def test_retry_after_in_milliseconds_takes_precedence():
    assert retry_after_seconds({"retry-after-ms": "250", "retry-after": "9"}) == 0.25


def test_retry_after_as_http_date():
    headers = {"retry-after": "Sun, 08 Nov 2026 12:00:30 GMT"}

    assert retry_after_seconds(headers, now=NOW) == 30.0


def test_retry_delay_in_gemini_error_body():
    body = '[{"error": {"code": 429, "details": [{"retryDelay": "41s"}]}}]'

    assert retry_after_seconds({}, body) == 41.0


def test_no_retry_hint_returns_none():
    assert retry_after_seconds({}, "erro qualquer") is None
    assert retry_after_seconds({"retry-after": "amanhã"}) is None


def test_breaker_opens_after_threshold_and_half_opens_after_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown_s=60, clock=clock)

    breaker.record_failure()
    assert breaker.state == "closed" and breaker.allow()

    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow()
    assert breaker.retry_in_s() == 60

    clock.now += 60
    assert breaker.state == "half_open" and breaker.allow()

    breaker.record_failure()  # falhou de novo no teste: volta a esperar
    assert breaker.state == "open"


def test_breaker_success_resets():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown_s=60, clock=clock)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()

    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 1


def test_hard_failure_opens_breaker_at_once():
    breaker = CircuitBreaker(threshold=5, cooldown_s=60, clock=FakeClock())

    breaker.record_failure(hard=True)

    assert breaker.state == "open"
