"""Reading from Kappa survives a hung TLS handshake.

Measured against the live server on 2026-10-12, three identical requests
for one dataset's entities:

    attempt 1: TLS 33.4s, response 37.9s  -> HTTP 200
    attempt 2: TLS  1.2s, no response     -> timeout
    attempt 3: TLS  0.13s, response 1.5s  -> HTTP 200

The hang is in the handshake, and when the handshake is fast the whole
request is fast. So waiting longer is the wrong fix — a new connection is
the fix. The flat 15s timeout did neither: it let a hung handshake eat the
whole budget and then gave up for good. 27 of 32 listing calls failed that
way, which the validation tab showed as "не удалось загрузить список".
"""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import kappa_client


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _Client:
    """One AsyncClient whose GET replays a scripted list of outcomes."""

    def __init__(self, outcomes, log):
        self.outcomes = outcomes
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        self.log.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install(monkeypatch, outcomes):
    """Each AsyncClient(...) call gets a fresh client — a retry must open a
    new connection, which is the whole point."""
    log = []
    captured = {}

    def _factory(**kw):
        captured.setdefault("timeout", kw.get("timeout"))
        return _Client(outcomes, log)

    monkeypatch.setattr(kappa_client.httpx, "AsyncClient", _factory)
    monkeypatch.setattr(kappa_client.asyncio, "sleep", _no_sleep)
    return log, captured


async def _no_sleep(_seconds):
    return None


@pytest.mark.asyncio
async def test_a_hung_handshake_is_retried_and_the_data_arrives(monkeypatch):
    """The exact sequence measured: a timeout, then a good response."""
    log, _ = _install(monkeypatch, [
        httpx.ConnectTimeout("handshake hung"),
        _Resp(200, [{"dsEntityId": "e1"}]),
    ])

    out = await kappa_client.get_dataset_entities("tok", 26, 3, 351)

    assert out == [{"dsEntityId": "e1"}]
    assert len(log) == 2, "повтора не было"


@pytest.mark.asyncio
async def test_a_first_try_success_does_not_retry(monkeypatch):
    """A healthy Kappa must not pay for this."""
    log, _ = _install(monkeypatch, [_Resp(200, [])])

    out = await kappa_client.get_dataset_entities("tok", 26, 3, 351)

    assert out == []
    assert len(log) == 1


@pytest.mark.asyncio
async def test_giving_up_returns_none_rather_than_raising(monkeypatch):
    """Callers treat None as 'ask again later'; an exception here would take
    down the whole validation tab."""
    log, _ = _install(monkeypatch, [
        httpx.ConnectTimeout("hung"),
        httpx.ReadTimeout("hung"),
        httpx.ConnectTimeout("hung"),
    ])

    out = await kappa_client.get_dataset_entities("tok", 26, 3, 351)

    assert out is None
    assert len(log) == kappa_client.READ_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_an_http_answer_is_not_retried(monkeypatch):
    """403 or 404 is an answer, not a hang. Retrying it changes nothing and
    triples how long the operator waits for the same refusal."""
    log, _ = _install(monkeypatch, [_Resp(404, None, "not found")])

    out = await kappa_client.get_dataset_entities("tok", 26, 3, 351)

    assert out is None
    assert len(log) == 1, "ответ сервера повторять нельзя"


@pytest.mark.asyncio
async def test_the_handshake_is_capped_shorter_than_the_read(monkeypatch):
    """The shape that makes the retry work. A flat timeout lets a hung
    handshake consume the entire budget, so the retry never gets a turn
    inside a sensible wait."""
    _, captured = _install(monkeypatch, [_Resp(200, [])])

    await kappa_client.get_dataset_entities("tok", 26, 3, 351)

    t = captured["timeout"]
    assert isinstance(t, httpx.Timeout), "нужен раздельный таймаут, не одно число"
    assert t.connect < t.read, (
        f"connect {t.connect} не короче read {t.read} — зависшее "
        f"рукопожатие снова съест весь бюджет"
    )


@pytest.mark.asyncio
async def test_entity_details_gets_the_same_treatment(monkeypatch):
    """The validation tab depends on both calls; today's log has 27 failures
    of the listing and 3 of the details. Fixing one would leave the same
    symptom arriving from the other."""
    log, _ = _install(monkeypatch, [
        httpx.ConnectTimeout("hung"),
        _Resp(200, {"files": [{"fileId": "f1", "fileName": "a.nii.gz"}]}),
    ])

    out = await kappa_client.get_entity_details("tok", 26, 3, 351, "e1")

    assert out == {"files": [{"fileId": "f1", "fileName": "a.nii.gz"}]}
    assert len(log) == 2


# --- Two defects found by review of the first version of this helper -------

@pytest.mark.asyncio
async def test_a_non_json_body_returns_none_instead_of_raising(monkeypatch):
    """The contract is 'parsed JSON, or None'. The first version called
    .json() outside the try, so a 200 carrying HTML raised and escaped —
    and a degraded Kappa behind a proxy serves exactly that. Before the
    retry was introduced this case returned None, so it was a regression.
    """
    class _Html(_Resp):
        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    log, _ = _install(monkeypatch, [_Html(200, None, "<html>502 Bad Gateway</html>")])

    out = await kappa_client.get_dataset_entities("tok", 26, 3, 351)

    assert out is None
    assert len(log) == 1, "мусор в теле — это ответ, повторять его незачем"


@pytest.mark.asyncio
async def test_the_total_wait_is_bounded(monkeypatch):
    """Per-attempt timeouts alone do not bound the call: three attempts that
    each hang on the READ are 3x30s plus pauses, about 94s. The validation
    tab polls every 10 seconds, so those pile up — nine overlapping requests
    holding connections open against a server that is already struggling.
    """
    clock = {"t": 0.0}

    def _fake_monotonic():
        return clock["t"]

    async def _slow_sleep(_seconds):
        clock["t"] += 30.0          # stands in for an attempt that hung

    log = []

    def _factory(**kw):
        return _Client([httpx.ReadTimeout("hung")] * 10, log)

    monkeypatch.setattr(kappa_client.httpx, "AsyncClient", _factory)
    monkeypatch.setattr(kappa_client.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(kappa_client.asyncio, "sleep", _slow_sleep)

    out = await kappa_client.get_dataset_entities("tok", 26, 3, 351)

    assert out is None
    assert len(log) < kappa_client.READ_RETRY_ATTEMPTS, (
        f"сделано {len(log)} попыток — бюджет "
        f"{kappa_client.READ_TOTAL_BUDGET_SECONDS}s не остановил их"
    )
