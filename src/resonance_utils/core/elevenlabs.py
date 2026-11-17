"""ElevenLabs HTTP client shared by every soundscape type.

Auth from ELEVENLABS_API_KEY (never logged), base URL from ELEVENLABS_API_BASE, retry with
exponential backoff on 429/5xx and network errors, and one usage line per successful call
appended to the usage log (ELEVENLABS_USAGE_LOG, default cache/el/usage.jsonl), so the cost of
every type can be audited in one place.

Caching is deliberately NOT here: each type owns its cache policy (the inducer keeps numbered
takes per prompt in raw/; the energizer uses a content-addressed request cache, core/cache.py).
"""

import datetime
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

API_BASE_DEFAULT = "https://api.elevenlabs.io"
USAGE_LOG_DEFAULT = os.path.join("cache", "el", "usage.jsonl")


class APIError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"HTTP {code}: {msg}")
        self.code = code


def load_dotenv(path=".env"):
    """Set variables from a KEY=VALUE .env file (cwd) that aren't already in the environment."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                os.environ.setdefault(k, v.strip().strip("'\""))
    except OSError:
        pass


def api_key(required=True):
    if not os.environ.get("ELEVENLABS_API_KEY"):
        load_dotenv()
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if required and not key:
        raise SystemExit("ELEVENLABS_API_KEY is not set (or use --from-dir / --dry-run)")
    return key


def api_base():
    return os.environ.get("ELEVENLABS_API_BASE", API_BASE_DEFAULT).rstrip("/")


def request_hash(endpoint, body, query=None):
    blob = json.dumps(dict(endpoint=endpoint, body=body, query=query or {}), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def log_usage(endpoint, body, query=None, seconds=None, n_bytes=None, path=None):
    path = path or os.environ.get("ELEVENLABS_USAGE_LOG", USAGE_LOG_DEFAULT)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(dict(
                time=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                endpoint=endpoint, seconds=seconds, bytes=n_bytes,
                request=request_hash(endpoint, body, query))) + "\n")
    except OSError as e:                                # never fail a paid call over the log
        print(f"    (usage log not written: {e})")


def post(endpoint, body, query=None, accept="audio/*", key=None, base=None, timeout=180,
         retries=6, seconds=None, with_headers=False, data=None, content_type="application/json"):
    """POST to endpoint (e.g. '/v1/sound-generation'), return the response bytes (and headers).
    body is JSON-encoded unless raw `data` is given (e.g. multipart). seconds: audio length
    requested, recorded in the usage log."""
    key = key if key is not None else api_key()
    base = base or api_base()
    qs = "&".join(f"{k}={v}" for k, v in (query or {}).items())
    url = f"{base}{endpoint}" + (f"?{qs}" if qs else "")
    req = urllib.request.Request(url, data=data if data is not None else json.dumps(body).encode(),
                                 method="POST", headers={"xi-api-key": key, "Content-Type": content_type,
                                                         "Accept": accept})
    delay = 4.0
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
                headers = dict(r.headers.items())
            log_usage(endpoint, body, query, seconds, len(blob))
            return (blob, headers) if with_headers else blob
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 429 or e.code >= 500:
                print(f"    HTTP {e.code}, retrying in {delay:.0f}s ...")
                time.sleep(delay)
                delay *= 2
                continue
            raise APIError(e.code, msg) from None
        except urllib.error.URLError as e:
            print(f"    network error {e.reason}, retrying in {delay:.0f}s ...")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("API failed after retries")


def sound_generation(text, duration, influence, model, output_format, loop=True, key=None, base=None):
    """Text-to-sound-effects. Returns the audio bytes in output_format."""
    body = {"text": text, "duration_seconds": duration, "prompt_influence": influence,
            "loop": loop, "model_id": model}
    return post("/v1/sound-generation", body, dict(output_format=output_format),
                key=key, base=base, seconds=duration)


def pcm_unavailable(err):
    """True when an error means the plan can't return PCM (fall back to a compressed format)."""
    return isinstance(err, APIError) and err.code in (400, 402, 403, 422)


def multipart(fields, files):
    """Encode a multipart/form-data body. files: {name: (filename, bytes, mime)}."""
    import uuid
    b = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out += [f"--{b}", f'Content-Disposition: form-data; name="{k}"', "", str(v)]
    parts = "\r\n".join(out).encode() + (b"\r\n" if out else b"")
    for k, (fn, blob, mime) in files.items():
        parts += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\n'
                  f"Content-Type: {mime}\r\n\r\n").encode() + blob + b"\r\n"
    return parts + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"
