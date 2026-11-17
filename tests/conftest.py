import pytest


@pytest.fixture(autouse=True)
def _isolated_api_state(tmp_path, monkeypatch):
    """No test may touch the real request cache or usage log (they hold paid audio)."""
    monkeypatch.setenv("ELEVENLABS_CACHE_DIR", str(tmp_path / "el_cache"))
    monkeypatch.setenv("ELEVENLABS_USAGE_LOG", str(tmp_path / "usage.jsonl"))
