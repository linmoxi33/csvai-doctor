"""Provider behaviour: malformed replies, retries, and the JSON rescue path.

These tests exist because every one of these cases happened against a real
endpoint at least once, and each one originally crashed the command instead of
degrading.
"""

import pytest

from csvai.errors import ModelRefusal, ProviderError, ProviderUnavailable
from csvai.provider import ChatProvider, extract_json_object
from tests.fakes import FakeTransport, RawPayload, make_config, make_provider


def test_extract_plain_object():
    assert extract_json_object('{"a": 1}') == {'a': 1}


def test_extract_object_from_fences_ignores_braces_inside_strings():
    text = 'Sure!\n```json\n{"sql": "SELECT {x} FROM t", "n": 2}\n```\nDone.'
    assert extract_json_object(text) == {'sql': 'SELECT {x} FROM t', 'n': 2}


def test_extract_returns_none_for_prose():
    assert extract_json_object('I cannot do that, sorry.') is None


def test_extract_handles_unbalanced_brace():
    assert extract_json_object('{"a": {"b": 1}') is None


def test_extract_handles_empty():
    assert extract_json_object('') is None


def test_unavailable_without_key():
    cfg = make_config(provider='openai', api_key=None, base_url='https://x/v1')
    provider = ChatProvider(cfg, transport=FakeTransport([]))
    assert provider.available() is False
    with pytest.raises(ProviderUnavailable, match='no API key'):
        provider.complete_json('s', 'u')


def test_unavailable_when_disabled():
    cfg = make_config()
    cfg.values['enabled'] = False
    provider = ChatProvider(cfg, transport=FakeTransport([]))
    with pytest.raises(ProviderUnavailable, match='disabled'):
        provider.complete_json('s', 'u')


def test_reply_without_json_is_repaired_once():
    # First reply is prose, second (after the repair prompt) is valid JSON.
    provider, transport = make_provider(['Here you go: {"ok": true}'])
    assert provider.complete_json('s', 'u') == {'ok': True}


def test_reply_that_never_yields_json_raises_refusal():
    provider, transport = make_provider(['no json at all', 'still no json'])
    with pytest.raises(ModelRefusal, match='did not return JSON'):
        provider.complete_json('s', 'u')
    assert len(transport.calls) == 2


def test_malformed_payload_shape_raises_refusal():
    # The provider must receive the malformed envelope itself, not a well-formed
    # one that happens to contain these keys.
    provider, _ = make_provider([RawPayload({'unexpected': 'shape'})])
    with pytest.raises(ModelRefusal):
        provider.complete_json('s', 'u')


def test_retryable_error_is_retried_then_succeeds():
    from csvai.provider import _Retryable

    provider, transport = make_provider([_Retryable('boom'), {'ok': True}])
    assert provider.complete_json('s', 'u') == {'ok': True}
    assert len(transport.calls) == 2


def test_retry_budget_is_respected():
    from csvai.provider import _Retryable

    cfg = make_config(max_retries=1)
    transport = FakeTransport([_Retryable('a'), _Retryable('b')])
    provider = ChatProvider(cfg, transport=transport, sleep=lambda _s: None)
    with pytest.raises(ProviderError, match='after 2 attempts'):
        provider.complete_json('s', 'u')
    assert len(transport.calls) == 2


def test_non_retryable_error_fails_immediately():
    provider, transport = make_provider([ProviderError('HTTP 400: bad request')])
    with pytest.raises(ProviderError, match='HTTP 400'):
        provider.complete_json('s', 'u')
    assert len(transport.calls) == 1


def test_json_mode_rejection_retries_without_response_format():
    provider, transport = make_provider(
        [ProviderError('HTTP 400: response_format is not supported'), {'ok': True}]
    )
    assert provider.complete_json('s', 'u') == {'ok': True}
    assert 'response_format' in transport.calls[0]['body']
    assert 'response_format' not in transport.calls[1]['body']


def test_usage_is_tracked():
    provider, _ = make_provider([{'ok': True}])
    provider.complete_json('s', 'u')
    usage = provider.usage.as_dict()
    assert usage['calls'] == 1
    assert usage['prompt_tokens'] == 10


def test_authorization_header_only_when_key_present():
    provider, transport = make_provider([{'ok': True}])
    provider.complete_json('s', 'u')
    assert transport.calls[0]['headers']['Authorization'] == 'Bearer test-key-0123456789'


def test_cache_hit_skips_the_network(tmp_path):
    from csvai.cache import ResponseCache

    cache = ResponseCache(tmp_path)
    provider, transport = make_provider([{'ok': True, 'v': 1}])
    first = provider.complete_json('s', 'u', cache=cache)
    second = provider.complete_json('s', 'u', cache=cache)
    assert first == second == {'ok': True, 'v': 1}
    assert len(transport.calls) == 1
    assert provider.usage.cached_hits == 1
    assert provider.usage.calls == 1


def test_cache_key_is_model_sensitive(tmp_path):
    from csvai.cache import ResponseCache

    cache = ResponseCache(tmp_path)
    assert cache.key('m1', 's', 'u') != cache.key('m2', 's', 'u')
    assert cache.key('m1', 's', 'u') == cache.key('m1', 's', 'u')


def test_corrupt_cache_entry_is_treated_as_miss(tmp_path):
    from csvai.cache import ResponseCache

    cache = ResponseCache(tmp_path)
    key = cache.key('m', 's', 'u')
    path = cache._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{not json', encoding='utf-8')
    assert cache.get(key) is None


def test_cache_write_failure_is_silent(tmp_path, monkeypatch):
    from csvai.cache import ResponseCache

    cache = ResponseCache(tmp_path / 'ro')

    def boom(*args, **kwargs):
        raise OSError('read-only filesystem')

    monkeypatch.setattr('tempfile.mkstemp', boom)
    cache.set('key', {'ok': True})  # must not raise
