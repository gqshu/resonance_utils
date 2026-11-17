"""Content-addressed request cache (energizer plan §5.1): cache/el/<sha1(request)>/ holds the
response bytes and its metadata, so an identical request never hits the API twice."""

import json
import os

from .elevenlabs import request_hash

CACHE_ROOT_DEFAULT = os.path.join("cache", "el")


def cached_call(endpoint, body, query, fetch, root=None, no_cache=False, suffix=".bin"):
    """Return (bytes, path, hit). fetch() makes the real call on a miss (or with no_cache).
    root defaults to ELEVENLABS_CACHE_DIR, else CACHE_ROOT_DEFAULT (looked up at call time)."""
    root = root or os.environ.get("ELEVENLABS_CACHE_DIR") or CACHE_ROOT_DEFAULT
    h = request_hash(endpoint, body, query)
    d = os.path.join(root, h)
    blob_path, meta_path = os.path.join(d, "response" + suffix), os.path.join(d, "request.json")
    if not no_cache and os.path.isfile(blob_path):
        with open(blob_path, "rb") as fh:
            return fh.read(), blob_path, True
    blob = fetch()
    os.makedirs(d, exist_ok=True)
    with open(blob_path + ".tmp", "wb") as fh:
        fh.write(blob)
    os.replace(blob_path + ".tmp", blob_path)
    with open(meta_path, "w") as fh:
        json.dump(dict(endpoint=endpoint, body=body, query=query or {}), fh, indent=2)
    return blob, blob_path, False
