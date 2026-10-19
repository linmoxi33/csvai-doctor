"""A small on-disk cache for model replies.

Two reasons this exists rather than being an optimisation detail:

* Cost. Classifying the same 60 columns while iterating on a prompt is a
  pointless bill; a content-addressed cache makes iteration free.
* Determinism. A cached answer replays byte-identically, so a test that
  exercises the AI path does not depend on the model's mood.

The key covers model, prompt, and payload - but never the API key, so two
accounts share a cache safely.
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

#: Bump when prompt structure changes in a way that invalidates old answers.
SCHEMA_VERSION = 1


class ResponseCache:
    def __init__(self, directory, ttl=30 * 24 * 3600):
        self.directory = Path(directory)
        self.ttl = ttl

    def key(self, model, system, user, hint=None):
        digest = hashlib.sha256()
        for part in (str(SCHEMA_VERSION), model or '', system or '', user or '', hint or ''):
            digest.update(part.encode('utf-8'))
            digest.update(b'\x00')
        return digest.hexdigest()

    def _path(self, key):
        return self.directory / key[:2] / f'{key}.json'

    def get(self, key):
        path = self._path(key)
        try:
            raw = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return None
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if self.ttl and time.time() - record.get('created', 0) > self.ttl:
            return None
        value = record.get('value')
        return value if isinstance(value, dict) else None

    def set(self, key, value):
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({'created': time.time(), 'value': value}, ensure_ascii=False)
            # Atomic write: a half-written cache file would be read back as a
            # corrupt answer, which is worse than a miss.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    handle.write(payload)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            # A read-only or full disk must never break the command.
            return

    def clear(self):
        removed = 0
        if not self.directory.exists():
            return removed
        for path in self.directory.rglob('*.json'):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def stats(self):
        if not self.directory.exists():
            return {'entries': 0, 'bytes': 0}
        entries = size = 0
        for path in self.directory.rglob('*.json'):
            try:
                size += path.stat().st_size
                entries += 1
            except OSError:
                pass
        return {'entries': entries, 'bytes': size}
