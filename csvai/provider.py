"""A single, dependency-free OpenAI-compatible chat client.

Why hand-rolled ``urllib`` instead of the official SDK? csvkit is a tool people
install globally; adding ``openai`` (plus its httpx/anyio tree) to an optional
extra that most users never enable is a poor trade. The wire format we need is
four fields wide and stable, so the SDK buys us nothing here.

Everything that can go wrong is translated into :class:`ProviderError` so the
caller has exactly one exception family to fall back from.
"""

import json
import random
import time
import urllib.error
import urllib.request

from .errors import ModelRefusal, ProviderError, ProviderUnavailable

#: Endpoints known to reject ``response_format``. We retry once without it
#: instead of failing, because a schema-less reply is still parseable when we
#: extract the first JSON object ourselves.
_JSON_MODE_UNSUPPORTED_HINTS = (
    'response_format',
    'json_object',
    'unsupported',
    'unknown field',
    'invalid_request_error',
)


class Usage:
    """Token accounting for one session, so cost is visible rather than implied."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_hits = 0
        self.seconds = 0.0

    def add(self, payload):
        self.calls += 1
        usage = payload.get('usage') or {}
        self.prompt_tokens += int(usage.get('prompt_tokens') or 0)
        self.completion_tokens += int(usage.get('completion_tokens') or 0)

    def as_dict(self):
        return {
            'calls': self.calls,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'cache_hits': self.cached_hits,
            'seconds': round(self.seconds, 2),
        }


class ChatProvider:
    """Talks to any ``/chat/completions`` endpoint.

    ``transport`` is injectable purely so tests can drive every failure branch
    without network access; production always uses :meth:`_urllib_transport`.
    """

    def __init__(self, cfg, transport=None, sleep=None):
        self.cfg = cfg
        self.usage = Usage()
        self._transport = transport or self._urllib_transport
        self._sleep = sleep or time.sleep

    # -- public ---------------------------------------------------------
    def available(self):
        from .config import is_configured

        return is_configured(self.cfg)

    def complete_json(self, system, user, schema_hint=None, cache=None):
        """Ask for a JSON object and return it parsed.

        *cache*, when given, is a :class:`~csvai.cache.ResponseCache`; a hit
        short-circuits the network entirely and is recorded in usage so the
        saving is measurable.
        """
        if not self.available():
            raise ProviderUnavailable(self._unavailable_reason())

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]
        key = None
        if cache is not None:
            key = cache.key(self.cfg.model, system, user, schema_hint)
            hit = cache.get(key)
            if hit is not None:
                self.usage.cached_hits += 1
                return hit

        text = self._chat(messages)
        data = extract_json_object(text)
        if data is None:
            # One repair attempt, cheaper than making the caller re-ask.
            text = self._chat(
                messages
                + [
                    {'role': 'assistant', 'content': text[:4000]},
                    {
                        'role': 'user',
                        'content': 'Return ONLY a single valid JSON object. No prose, no markdown fences.',
                    },
                ]
            )
            data = extract_json_object(text)
        if data is None:
            raise ModelRefusal(f'model did not return JSON: {text[:200]!r}')

        if key is not None:
            cache.set(key, data)
        return data

    # -- internals ------------------------------------------------------
    def _unavailable_reason(self):
        if not self.cfg.enabled:
            return 'AI layer disabled (CSVAI_DISABLE or enabled=false)'
        if not self.cfg.base_url:
            return 'no base_url configured'
        if not self.cfg.model:
            return 'no model configured'
        return 'no API key configured for a remote provider'

    def _chat(self, messages):
        body = {
            'model': self.cfg.model,
            'messages': messages,
            'temperature': self.cfg['temperature'],
            'response_format': {'type': 'json_object'},
        }
        try:
            payload = self._post_with_retries(body)
        except ProviderError as exc:
            if not _looks_like_json_mode_rejection(exc):
                raise
            # Retry once without response_format. Copy the body first: mutating
            # the dict in place also rewrites the request that was already sent,
            # which makes any transport log lie about what happened.
            fallback = {k: v for k, v in body.items() if k != 'response_format'}
            payload = self._post_with_retries(fallback)

        self.usage.add(payload)
        try:
            return payload['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError):
            raise ModelRefusal(f'unexpected response shape: {str(payload)[:200]}') from None

    def _post_with_retries(self, body):
        url = f'{self.cfg.base_url}/chat/completions'
        headers = {'Content-Type': 'application/json'}
        if self.cfg.api_key:
            headers['Authorization'] = f'Bearer {self.cfg.api_key}'

        attempts = int(self.cfg['max_retries']) + 1
        last = None
        for attempt in range(attempts):
            started = time.time()
            try:
                payload = self._transport(url, headers, body, float(self.cfg['timeout']))
                self.usage.seconds += time.time() - started
                return payload
            except _Retryable as exc:
                self.usage.seconds += time.time() - started
                last = exc
                if attempt == attempts - 1:
                    break
                # Exponential backoff with jitter: keeps a flaky endpoint from
                # turning into a synchronised retry storm.
                delay = min(8.0, 0.5 * (2**attempt)) * (0.5 + random.random())
                self._sleep(delay)
            except ProviderError as exc:
                self.usage.seconds += time.time() - started
                raise exc
        raise ProviderError(f'{last} (after {attempts} attempts)')

    @staticmethod
    def _urllib_transport(url, headers, body, timeout):
        data = json.dumps(body).encode('utf-8')
        request = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = ''
            try:
                detail = exc.read().decode('utf-8', 'replace')[:400]
            except Exception:  # pragma: no cover - body already consumed
                pass
            message = f'HTTP {exc.code}: {detail or exc.reason}'
            if exc.code in (429, 500, 502, 503, 504):
                raise _Retryable(message) from None
            raise ProviderError(message) from None
        except urllib.error.URLError as exc:
            raise _Retryable(f'connection failed: {exc.reason}') from None
        except TimeoutError:
            raise _Retryable(f'timed out after {timeout}s') from None
        try:
            return json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError(f'endpoint returned non-JSON: {raw[:200]!r}') from None


class _Retryable(ProviderError):
    """Marker: worth trying again (transport hiccup or 5xx/429)."""


def _looks_like_json_mode_rejection(exc):
    text = str(exc).lower()
    return any(hint in text for hint in _JSON_MODE_UNSUPPORTED_HINTS)


def extract_json_object(text):
    """Best-effort JSON object extraction from a chat reply.

    Models wrap JSON in fences, prepend "Sure!", or emit a trailing sentence.
    We accept the outermost balanced ``{...}`` block, ignoring braces inside
    strings, which is the failure mode a naive ``find('{')`` gets wrong.

    If the outermost object is unbalanced - truncated output, for instance - we
    return ``None`` rather than salvaging an inner object. A partial object
    would parse cleanly but describe only part of the answer, and a caller
    cannot tell the difference. Failing that check is what triggers the single
    repair attempt upstream.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith('{'):
        try:
            value = json.loads(stripped)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

    start = stripped.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    # Never closed: refuse rather than guess which inner object was meant.
    return None
