"""A 429 from BIIGLE is waited out a bounded number of times, then the run stops.

The account was locked on 2026-10-16 after a loop carried on through 429s. The
request layer now honours Retry-After, re-sends at most `max_retries` times,
and after giving up sends nothing more for the rest of the process.
"""

import concurrent.futures

import pytest
import requests

from spyfish.biigle import biigle_handler as bh
from spyfish.biigle.biigle_handler import (
    BiigleHandler,
    BiigleRateLimitError,
    _ApiWithTimeout,
    _RateLimit,
)
from spyfish.config.biigle import BiigleConfig
from spyfish.config.wrapper import config


class _Resp:
    def __init__(self, status=200, headers=None, payload=None):
        self.status_code, self.headers, self._payload = status, headers or {}, payload
        self.content = b"x"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)


class _Wire:
    """A requests method: answers from a script, then repeats the last answer."""

    def __init__(self, *responses):
        self.responses, self.sent = list(responses), 0

    def __call__(self, url, *args, **kwargs):
        r = self.responses[min(self.sent, len(self.responses) - 1)]
        self.sent += 1
        return r


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Fake time: sleeps are recorded and advance the clock, nothing waits."""
    _RateLimit.reset()
    state = {"now": 1000.0, "slept": []}

    def sleep(secs):
        state["slept"].append(secs)
        state["now"] += secs

    monkeypatch.setattr(bh.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(bh.time, "sleep", sleep)
    yield state
    _RateLimit.reset()


def _api(max_retries=2):
    return _ApiWithTimeout(
        "a@b.nz", "token", max_retries=max_retries, fallback_wait=60, max_wait=300
    )


def test_429_waits_retry_after_then_succeeds(clock):
    wire = _Wire(_Resp(429, {"Retry-After": "7"}), _Resp(200, payload={"ok": 1}))
    resp = _api().call(wire, "volumes/5/filenames")
    assert resp.json() == {"ok": 1}
    assert wire.sent == 2
    assert clock["slept"] == [7.0]


def test_429_without_header_uses_the_fallback_wait(clock):
    wire = _Wire(_Resp(429), _Resp(200))
    _api().call(wire, "projects/1/volumes")
    assert clock["slept"] == [60.0]


def test_persistent_429_is_bounded_then_raises_and_stops_everything(clock):
    wire = _Wire(_Resp(429, {"Retry-After": "5"}))
    api = _api(max_retries=2)

    with pytest.raises(BiigleRateLimitError) as exc:
        api.call(wire, "images/1/labels")
    assert wire.sent == 3  # the request and two re-sends, never more
    assert clock["slept"] == [5.0, 5.0]
    assert exc.value.response.status_code == 429

    # any later request, from any handler, is refused without being sent
    other = _Wire(_Resp(200))
    with pytest.raises(BiigleRateLimitError):
        _api().call(other, "projects/1/volumes")
    assert other.sent == 0


def test_retry_after_longer_than_the_cap_raises_without_sleeping(clock):
    wire = _Wire(_Resp(429, {"Retry-After": "3600"}))
    with pytest.raises(BiigleRateLimitError):
        _api().call(wire, "images/1/labels")
    assert wire.sent == 1
    assert clock["slept"] == []


def test_429_raises_even_when_the_caller_reads_statuses_itself(clock):
    wire = _Wire(_Resp(429, {"Retry-After": "1"}))
    with pytest.raises(BiigleRateLimitError):
        _api(max_retries=1).call(wire, "reports/9", False)
    assert wire.sent == 2


def test_other_errors_are_not_retried(clock):
    wire = _Wire(_Resp(500))
    with pytest.raises(requests.exceptions.HTTPError) as exc:
        _api().call(wire, "projects/1/volumes")
    assert not isinstance(exc.value, BiigleRateLimitError)
    assert wire.sent == 1
    assert clock["slept"] == []


def test_a_pause_set_by_one_request_holds_back_the_next(clock):
    api = _api()
    api.call(_Wire(_Resp(429, {"Retry-After": "30"}), _Resp(200)), "a")
    _RateLimit.resume_at = clock["now"] + 12  # as another thread's 429 would
    clock["slept"].clear()
    api.call(_Wire(_Resp(200)), "b")
    assert clock["slept"] == [12.0]


def test_retry_after_as_http_date(monkeypatch):
    class _Now(bh.datetime):
        @classmethod
        def now(cls, tz=None):
            return bh.datetime(2026, 11, 11, 0, 0, 0, tzinfo=bh.timezone.utc)

    monkeypatch.setattr(bh, "datetime", _Now)
    resp = _Resp(429, {"Retry-After": "Wed, 11 Nov 2026 00:00:40 GMT"})
    assert bh._retry_after_secs(resp) == 40.0
    assert bh._retry_after_secs(_Resp(429, {"Retry-After": "soon"})) is None


# ── attach_image_labels ──────────────────────────────────────────────────


class _LabelApi:
    """Accepts `ok` label POSTs, then is rate limited for good."""

    def __init__(self, ok):
        self.ok, self.posts = ok, 0

    def post(self, url, json=None):
        self.posts += 1
        if self.posts > self.ok:
            raise BiigleRateLimitError("429", response=_Resp(429))
        return _Resp(200, payload={"id": self.posts})


def _handler(api):
    h = BiigleHandler.__new__(BiigleHandler)  # skip auth/network init
    h.api = api
    return h


def _workers(monkeypatch, n):
    monkeypatch.setattr(type(config), "image_label_workers", property(lambda self: n))


def test_rate_limit_stops_the_remaining_labels(monkeypatch):
    _workers(monkeypatch, 1)
    api = _LabelApi(ok=2)
    with pytest.raises(BiigleRateLimitError):
        _handler(api).attach_image_labels([(i, 900) for i in range(50)])
    assert api.posts == 3  # two attached, the one that was refused, then none


def test_rate_limit_stop_holds_with_several_threads(monkeypatch):
    _workers(monkeypatch, 4)
    api = _LabelApi(ok=2)
    with pytest.raises(BiigleRateLimitError):
        _handler(api).attach_image_labels([(i, 900) for i in range(200)])
    # at most the requests already in flight on the other threads
    assert api.posts <= 3 + 4


def test_other_label_failures_are_counted_not_raised(monkeypatch):
    _workers(monkeypatch, 2)

    class _Api:
        def post(self, url, json=None):
            if url == "images/2/labels":
                raise Exception("The image already has this label.")
            return _Resp(200, payload={"id": 1})

    assert _handler(_Api()).attach_image_labels([(1, 9), (2, 9), (3, 9)]) == 2


def test_worker_count_comes_from_config(monkeypatch):
    _workers(monkeypatch, 3)
    seen = {}
    real = concurrent.futures.ThreadPoolExecutor

    def spy(max_workers=None, **kwargs):
        seen["max_workers"] = max_workers
        return real(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", spy)
    _handler(_LabelApi(ok=10)).attach_image_labels([(1, 9), (2, 9)])
    assert seen["max_workers"] == 3


def test_volume_polling_does_not_continue_through_a_rate_limit(monkeypatch):
    h = _handler(None)
    calls = []

    def get_volumes(project_id=None):
        calls.append(project_id)
        raise BiigleRateLimitError("429", response=_Resp(429))

    monkeypatch.setattr(h, "get_volumes", get_volumes)
    with pytest.raises(BiigleRateLimitError):
        h.resolve_real_volume_id("KSF_20240124_BUV ML", 1)
    assert len(calls) == 1


# ── config ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "key",
    [
        "image_label_workers",
        "rate_limit_max_retries",
        "rate_limit_fallback_wait_secs",
        "rate_limit_max_wait_secs",
    ],
)
def test_missing_config_key_raises(key):
    cfg = BiigleConfig.__new__(BiigleConfig)
    cfg._yaml_config = {"biigle": {}}
    with pytest.raises(KeyError, match=key):
        getattr(cfg, key)


def test_zero_workers_is_refused():
    cfg = BiigleConfig.__new__(BiigleConfig)
    cfg._yaml_config = {"biigle": {"image_label_workers": 0}}
    with pytest.raises(ValueError):
        cfg.image_label_workers
