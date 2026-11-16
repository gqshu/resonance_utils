import io
import json
import urllib.error

import numpy as np
import pytest

from resonance_utils.core import cache, elevenlabs as el, loop, slots


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_post_retries_then_logs_usage(monkeypatch, tmp_path):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b""))
        return _Resp(b"audio")

    monkeypatch.setattr(el.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(el.time, "sleep", lambda s: None)
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ELEVENLABS_USAGE_LOG", str(log))
    blob = el.sound_generation("rain", 30, 0.6, "m", "pcm_48000", key="k", base="http://x")
    assert blob == b"audio" and len(calls) == 2
    assert calls[0] == "http://x/v1/sound-generation?output_format=pcm_48000"
    row = json.loads(log.read_text())
    assert row["endpoint"] == "/v1/sound-generation" and row["seconds"] == 30 and "k" not in json.dumps(row)


def test_client_error_is_typed(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 402, "pay", {}, io.BytesIO(b"no pcm"))

    monkeypatch.setattr(el.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(el.APIError) as e:
        el.post("/v1/x", {}, key="k", base="http://x")
    assert el.pcm_unavailable(e.value) and str(e.value).startswith("HTTP 402:")


def test_request_cache(tmp_path):
    n = []
    fetch = lambda: n.append(1) or b"r"
    a = cache.cached_call("/v1/x", {"a": 1}, None, fetch, root=str(tmp_path))
    b = cache.cached_call("/v1/x", {"a": 1}, None, fetch, root=str(tmp_path))
    c = cache.cached_call("/v1/x", {"a": 2}, None, fetch, root=str(tmp_path))
    assert (a[2], b[2], c[2]) == (False, True, False) and len(n) == 2 and b[0] == b"r"


def test_close_loop_is_exact():
    x = np.random.default_rng(0).standard_normal((48000 * 3, 2))
    y = loop.close_loop(x, 48000 * 2, 4800)
    assert y.shape == (96000, 2)


def test_slots():
    assert slots.clean_carriers([440, 0, 110]) == [110, 440]
    assert slots.near_carrier(225, slots.DEFAULT_CARRIERS_HZ) == [220.0]
